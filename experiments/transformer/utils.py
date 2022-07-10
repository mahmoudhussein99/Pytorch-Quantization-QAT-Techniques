import torch

import fairseq
from copy import deepcopy
from opt_quant import *
from experiments.utils import get_q_modules, get_last_layer_modules, get_mha_modules, get_fc_q_modules, get_ln_modules, get_mha_linear_modules
from opt_quant import QMultiheadAttention, QLayerNorm


def tformer_replace_layer_norm(args, model, module_name=''):
    ln_act_qmodule = get_ln_modules(args)

    for n, module in model.named_children():

        if len(list(module.children())) > 0:
            ## compound module, go inside it
            tformer_replace_layer_norm(args, module, module_name=module_name+'{} '.format(n))

        if 'layer_norm' in n:
            quantizedLayerNorm = QLayerNorm(
                normalized_shape=module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                act_qmodule=deepcopy(ln_act_qmodule),
            )
            setattr(model, n, deepcopy(quantizedLayerNorm))



def tformer_replace_linear(args, model, module_name=''):

    last_w_qmodule, last_act_qmodule, last_err_qmodule = get_last_layer_modules(args)
    mha_linear_w_qmodule, mha_linear_act_qmodule, mha_linear_err_qmodule = get_mha_linear_modules(args)
    fc_w_qmodule, fc_act_qmodule, fc_err_qmodule = get_fc_q_modules(args)

    fc2_act_qmodule = deepcopy(fc_act_qmodule)
    if not isinstance(fc2_act_qmodule, nn.Identity):
        fc2_act_qmodule.mode = 'unsigned' #to use instead of ReLU

    # import pdb; pdb.set_trace()
    for n, module in model.named_children():

        if len(list(module.children())) > 0:
            ## compound module, go inside it
            tformer_replace_linear(args, module, module_name=module_name+'{} '.format(n))

        # return
        # Remove Activations when we use PACT
        if isinstance(module, nn.ReLU) and 'pact' in args.act_qmode :
            setattr(model, n, nn.Identity())

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                bias_status = True
            else:
                bias_status = False

            if 'output_projection' in n:
                quantizedLinear = QLinear(
                    in_features=module.in_features, out_features=module.out_features,
                    bias=bias_status,
                    w_qmodule=deepcopy(last_w_qmodule),
                    act_qmodule=deepcopy(last_act_qmodule),
                    err_qmodule=deepcopy(last_err_qmodule),
                )
            elif 'fc' in n:
                if 'fc2' in n:
                    quantizedLinear = QLinear(
                        in_features=module.in_features, out_features=module.out_features,
                        bias=bias_status,
                        w_qmodule=deepcopy(fc_w_qmodule),
                        act_qmodule=deepcopy(fc2_act_qmodule),
                        err_qmodule=deepcopy(fc_err_qmodule),
                    )
                else:
                    quantizedLinear = QLinear(
                        in_features=module.in_features, out_features=module.out_features,
                        bias=bias_status,
                        w_qmodule=deepcopy(fc_w_qmodule),
                        act_qmodule=deepcopy(fc_act_qmodule),
                        err_qmodule=deepcopy(fc_err_qmodule),
                    )
            else:
                quantizedLinear = QLinear(
                    in_features=module.in_features, out_features=module.out_features,
                    bias=bias_status,
                    w_qmodule=deepcopy(mha_linear_w_qmodule),
                    act_qmodule=deepcopy(mha_linear_act_qmodule),
                    err_qmodule=deepcopy(mha_linear_err_qmodule),
                )
            setattr(model, n, deepcopy(quantizedLinear))


def tformer_replace_mha(args, model: torch.nn.Module, module_name=''):

    act_qmodule, err_qmodule = get_mha_modules(args)

    for n, module in model.named_children():
        if isinstance(module, fairseq.modules.MultiheadAttention) or isinstance(module, torch.nn.MultiheadAttention):
            q_matmul_act = deepcopy(act_qmodule)
            k_matmul_act = deepcopy(act_qmodule)
            v_matmul_act = deepcopy(act_qmodule)
            smax_matmul_act = deepcopy(act_qmodule)

            k_q_matmul_err_quant = deepcopy(err_qmodule)
            smax_v_matmul_err_quant = deepcopy(err_qmodule)

            if module.k_proj.bias is not None:
                bias_status = True
            else:
                bias_status = False

            if module.bias_k is None:
                add_bias_kv = False
            else:
                add_bias_kv = True

            quantizedMHA = QMultiheadAttention(
                embed_dim=module.embed_dim, num_heads=module.num_heads,
                kdim=module.kdim, vdim=module.vdim, dropout=module.dropout_module.p,
                bias=bias_status, add_bias_kv=add_bias_kv,
                add_zero_attn=module.add_zero_attn,
                self_attention=module.self_attention,
                encoder_decoder_attention=module.encoder_decoder_attention,
                q_noise=args.quant_noise_pq,
                qn_block_size=args.quant_noise_pq_block_size,
                q_matmul_act=q_matmul_act,
                k_matmul_act=k_matmul_act,
                v_matmul_act=v_matmul_act,
                smax_matmul_act=smax_matmul_act,
                k_q_matmul_error=k_q_matmul_err_quant,
                smax_v_matmul_error=smax_v_matmul_err_quant,
            )

            setattr(model, n, quantizedMHA)

        elif len(list(module.children())) > 0:
            ## compound module, go inside it
            tformer_replace_mha(args, module, module_name=module_name+'{} '.format(n))

