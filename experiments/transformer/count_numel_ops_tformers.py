import logging
import os
import sys

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger()

import pandas as pd
import numpy as np
from torch.nn import Linear, LayerNorm, Embedding
from fairseq import (
    options,
    tasks,
    utils,
)
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.modules import MultiheadAttention, SinusoidalPositionalEmbedding
from fairseq.models import register_model_architecture

@register_model_architecture("transformer", "tiny_transformer_v2")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    # return base_architecture(args)



def forward_hook(module, data_in, data_out):
    setattr(module, 'act', data_in[0])


def backward_hook(module, grad_in, grad_out):
    setattr(module, 'err', grad_out[0])


def main(cfg: FairseqConfig) -> None:
    assert cfg.dataset.max_tokens is not None, "Please specify --max-tokens"
    assert cfg.criterion, "Please specify criterion to train a model"

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    # Build model and criterion
    model = task.build_model(cfg.model)
    model.prepare_for_onnx_export_()
    criterion = task.build_criterion(cfg.criterion)

    logger.info(f'num params: {sum(p.numel() for p in model.parameters())}')

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (LayerNorm, Linear, Embedding, SinusoidalPositionalEmbedding)):
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    # Load dataset
    logger.info('loading dataset')
    task.load_dataset('train')

    batch_itr = task.get_batch_iterator(task.dataset('train'), max_tokens=cfg.dataset.max_tokens)
    batch = next(batch_itr.next_epoch_itr())
    model.train()
    model.zero_grad()
    loss, _, _ = criterion(model, batch)
    loss.backward()

    # Count numel & ops
    keys = ['type', 'weight', 'bias', 'act', 'err', 'ops_w_a', 'ops_w_e', 'ops_a_e', 'ops_w_g', 'ops_a_a', 'ops_smax_drop']
    counts = {key: [] for key in keys}
    names = []

    for name, module in model.named_modules():
        if isinstance(module, (LayerNorm, MultiheadAttention)) \
                or 'fc' in name \
                or 'decoder.output_projection' in name \
                or 'embed_tokens' in name \
                or 'embed_positions' in name:


            # collect numel
            if isinstance(module, (LayerNorm, Linear)):

                if isinstance(module, Linear):
                    counts['type'].append('Linear')
                else:
                    counts['type'].append('Normalization')

                counts['weight'].append(module.weight.numel())
                if module.bias is not None:
                    counts['bias'].append(module.bias.numel())
                else:
                    counts['bias'].append(0)
                counts['act'].append(module.act.numel())
                counts['err'].append(module.err.numel())
            elif isinstance(module, MultiheadAttention):

                w = b = a = e = 0
                mha_lin_w = mha_lin_b = mha_lin_a = mha_lin_e = 0
                for attr in ['k_proj', 'v_proj', 'q_proj', 'out_proj']:
                    linear = getattr(module, attr)
                    mha_lin_w += linear.weight.numel()
                    mha_lin_b += linear.bias.numel()
                    mha_lin_e += linear.err.numel()
                mha_lin_a += module.out_proj.act.numel()
                mha_lin_a += module.q_proj.act.numel()  # same act is used for k and v
                if name == 'decoder.layers.0.encoder_attn':
                    mha_lin_a += module.q_proj.act.numel()  # encoder output
                seq, bs, h_dim = module.k_proj.act.size()
                query_size = seq * bs * h_dim

                a += query_size * 3  # q, k, v
                a += bs * seq * seq  # attn_weights
                a += bs * seq * seq  # attn_probs
                a += bs * seq * h_dim  # attn

                e += bs * seq * h_dim  # attn
                e += bs * seq * seq  # attn_probs
                e += bs * seq * seq  # attn_weights

                counts['type'].append("MHA_LINEAR")
                counts['weight'].append(mha_lin_w)
                counts['bias'].append(mha_lin_b)
                counts['act'].append(mha_lin_a)
                counts['err'].append(mha_lin_e)
                names.append(name + '_Linear')

                counts['type'].append('MHA')
                counts['weight'].append(w)
                counts['bias'].append(b)
                counts['act'].append(a)
                counts['err'].append(e)



            elif isinstance(module, Embedding):
                counts['type'].append('Embedding')
                counts['weight'].append(module.weight.numel())
                counts['bias'].append(0)
                counts['act'].append(module.act.numel())
                counts['err'].append(module.err.numel())
            else:
                assert isinstance(module, SinusoidalPositionalEmbedding)
                counts['type'].append('Pos_Embedding')
                counts['weight'].append(module.weights.numel())
                counts['bias'].append(0)
                counts['act'].append(0)
                counts['err'].append(0)
            names.append(name)
            # collect ops
            ops_w_e = ops_a_e = ops_w_a = ops_w_g = ops_a_a = ops_smax_drop = 0
            if isinstance(module, Linear):
                seq, bs, h_dim = module.act.size()
                f_out, f_in = module.weight.size()
                ops_w_a = ops_w_e = ops_a_e = seq * bs * f_out * f_in
                ops_w_g = module.weight.numel()  # weight upd
            elif isinstance(module, LayerNorm):
                seq, bs, h_dim = module.act.size()
                act_numel = seq * bs * h_dim
                ops_a_a = act_numel * 4  # normalization (mean, var, sub, div)
                ops_w_a = ops_w_e = act_numel * 2  # gamma, beta
                ops_a_e = act_numel  # gamma grad
                ops_a_e += act_numel * h_dim  # normalization grad
                ops_w_g = module.weight.numel() * 2  # gamma, beta upd
            elif isinstance(module, MultiheadAttention):
                seq, bs, h_dim = module.k_proj.act.size()
                mha_lin_ops_k_proj = mha_lin_ops_v_proj = mha_lin_ops_q_proj = seq * bs * h_dim * h_dim
                ops_matmul_qk = bs * seq * seq * h_dim
                ops_scale = bs * seq * seq
                ops_softmax = bs * seq * (seq * 2)  # sum, div
                ops_dropout = bs * seq * seq
                ops_matmul_attn_v = bs * seq * seq * h_dim
                mha_lin_ops_out_proj = seq * bs * h_dim * h_dim

                # MHA_Linear:
                mha_lin_ops_w_a = mha_lin_ops_w_e = sum([mha_lin_ops_q_proj,
                                         mha_lin_ops_k_proj,
                                         mha_lin_ops_v_proj,
                                         mha_lin_ops_out_proj])
                mha_lin_ops_a_a = 0
                mha_lin_ops_w_g = sum([module.q_proj.weight.numel(),
                               module.k_proj.weight.numel(),
                               module.v_proj.weight.numel(),
                               module.out_proj.weight.numel()])
                mha_lin_ops_a_e = sum([mha_lin_ops_q_proj,
                               mha_lin_ops_k_proj,
                               mha_lin_ops_v_proj,
                               mha_lin_ops_out_proj])

                counts['ops_w_e'].append(mha_lin_ops_w_e)
                counts['ops_a_e'].append(mha_lin_ops_a_e)
                counts['ops_w_a'].append(mha_lin_ops_w_a)
                counts['ops_a_a'].append(mha_lin_ops_a_a)
                counts['ops_w_g'].append(mha_lin_ops_w_g*10) #because of Adam
                counts['ops_smax_drop'].append(0)


                # SelfAttention:
                ops_w_a = ops_w_e = sum([0])
                ops_a_a = sum([ops_matmul_qk,
                               ops_scale,
                               ops_matmul_attn_v])

                ops_matmul_qk_bwd = (bs * seq * h_dim) * 2
                ops_scale_bwd = bs * seq * seq
                ops_softmax_bwd = bs * seq * (seq * 2)  # mul, sum
                ops_dropout_bwd = bs * seq * seq
                ops_matmul_attn_v_bwd = (bs * seq * seq * h_dim) * 2
                ops_a_e = sum([
                               ops_matmul_qk_bwd,
                               ops_scale_bwd,
                               ops_matmul_attn_v_bwd])
                ops_w_g = 0
                ops_smax_drop = sum([
                    ops_softmax_bwd,
                    ops_dropout_bwd,
                    ops_softmax,
                    ops_dropout,
                ])


            elif isinstance(module, Embedding):
                ops_w_g = module.weight.numel()
                _, h_dim = module.weight.shape
                bs, seq = module.act.shape
                ops_w_a = bs * seq
                ops_w_e = bs * seq * h_dim
            else:
                assert isinstance(module, SinusoidalPositionalEmbedding)
                ops_w_a = module.act.numel()


            counts['ops_w_e'].append(ops_w_e)
            counts['ops_a_e'].append(ops_a_e)
            counts['ops_w_a'].append(ops_w_a)
            counts['ops_a_a'].append(ops_a_a)
            counts['ops_w_g'].append(ops_w_g*10) #because of Adam
            counts['ops_smax_drop'].append(ops_smax_drop)


            if isinstance(module, LayerNorm):
                counts['type'].append('Shortcuts')
                names.append(name+'_shortcut_add')
                counts['ops_w_e'].append(0)
                counts['ops_a_e'].append(0)
                counts['ops_w_a'].append(0)
                counts['ops_a_a'].append(ops_a_a)
                counts['ops_w_g'].append(0)
                counts['ops_smax_drop'].append(0)

                counts['weight'].append(0)
                counts['bias'].append(0)
                counts['act'].append(0)
                counts['err'].append(0)

    # import pdb; pdb.set_trace()
    # print(counts)
    # print(names)
    df = pd.DataFrame(counts, index=names)
    print(df)
    df.to_csv(cfg.model.arch + '.csv')



if __name__ == "__main__":
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    main(cfg)
