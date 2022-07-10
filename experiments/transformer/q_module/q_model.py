from fairseq.models.transformer import base_architecture
from fairseq.models import register_model_architecture


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "qtransformer_base")
def transformer_vaswani_wmt_en_de_big(args):

    # remove the relu activations from the FC2
    # Check these lines:
    # https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/models/transformer/transformer_legacy.py#L188
    # https://github.com/pytorch/fairseq/blob/c71870f370455e6154c730e8822ea323b5f266f6/fairseq/modules/transformer_layer.py#L42
    # https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/utils.py#L540
    if args.fc_act_qmode != 'none':
        args.activation_fn = getattr(args, "activation_fn", "linear")
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "qtransformer_vaswani_wmt_en_de_big")
def qtransformer_vaswani_wmt_en_de_big(args):

    # remove the relu activations from the FC2
    # Check these lines:
    # https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/models/transformer/transformer_legacy.py#L188
    # https://github.com/pytorch/fairseq/blob/c71870f370455e6154c730e8822ea323b5f266f6/fairseq/modules/transformer_layer.py#L42
    # https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/utils.py#L540
    if args.fc_act_qmode != 'none':
        args.activation_fn = getattr(args, "activation_fn", "linear")

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)




@register_model_architecture("transformer", "tiny_transformer_v2")
def tiny_architecture(args):

    # remove the relu activations from the FC2
    # Check these lines:
    # https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/models/transformer/transformer_legacy.py#L188
    # https://github.com/pytorch/fairseq/blob/c71870f370455e6154c730e8822ea323b5f266f6/fairseq/modules/transformer_layer.py#L42
    # https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/utils.py#L540
    if args.fc_act_qmode != 'none':
        args.activation_fn = getattr(args, "activation_fn", "linear")

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    base_architecture(args)
