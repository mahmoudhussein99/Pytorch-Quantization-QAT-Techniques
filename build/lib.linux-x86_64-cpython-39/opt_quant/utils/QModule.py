import copy

import torch
import math
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction


from typing import Dict, Optional, Tuple
from fairseq import utils as fairseq_utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.layer_norm import LayerNorm
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import BatchNorm2d
from copy import deepcopy


import opt_quant


def bn(bn_type='BN', size=128,
       bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
       **kwargs):
    if bn_type == 'BN':
        return QBatchNorm2d(size, act_qmodule=bn_act_qmodule)
    elif bn_type == 'RangeBN':

        # Forward bits
        if isinstance(bn_act_qmodule, nn.Identity) or isinstance(bn_w_qmodule, nn.Identity):
            fwd_bits = 32
        else:
            fwd_bits = max(bn_act_qmodule.bitwidth, bn_w_qmodule.bitwidth)

        if isinstance(bn_err_qmodule, nn.Identity):
            bwd_bits = 32
        else:
            bwd_bits = bn_err_qmodule.bitwidth

        return RangeBN(size, num_bits=fwd_bits, num_bits_grad=bwd_bits)
    else:
        raise ValueError

class QLinear(nn.Linear):
    def __init__(self,
                 in_features, out_features, bias=True,
                 w_qmodule=nn.Identity(),
                 act_qmodule=nn.Identity(),
                 err_qmodule=nn.Identity()
                 ):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.w_quant = w_qmodule
        self.act_quant = act_qmodule
        self.err_quant = err_qmodule

    def forward(self, inp):
        #  Weight quantization
        quantized_w = self.w_quant(self.weight)

        # Activation Quantization
        quantized_inp = self.act_quant(inp)

        # Linear Operation
        output = F.linear(quantized_inp, quantized_w, self.bias)

        # Error Quantization (backward)
        return self.err_quant(output)


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 w_qmodule=nn.Identity(),
                 act_qmodule=nn.Identity(),
                 err_qmodule=nn.Identity()
                 ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)

        self.w_quant = w_qmodule
        self.act_quant = act_qmodule
        self.err_quant = err_qmodule

    def forward(self, inp):
        quantized_inp = self.act_quant(inp)

        #  Weight quantization
        quantized_w = self.w_quant(self.weight)

        # Linear Operation
        output = F.conv2d(quantized_inp, quantized_w, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)

        # Error Quantization (backward)
        return self.err_quant(output)


class QLayerNorm(nn.Module):
    def __init__(
            self,
            normalized_shape,
            eps=1e-5,
            elementwise_affine=True,
            export=False,
            act_qmodule=nn.Identity(),
    ) -> None:

        super(QLayerNorm, self).__init__()
        self.model = LayerNorm(normalized_shape, eps, elementwise_affine, export)
        self.act_quant = copy.deepcopy(act_qmodule)

    def forward(self, x):
        return self.model.forward(self.act_quant(x))


class QBatchNorm2d(BatchNorm2d):
    # This module only quantizes the input of the batch normalizaiton!
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None,
            act_qmodule=nn.Identity(),
    ) -> None:

        super(QBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.act_quant = copy.deepcopy(act_qmodule)

    def forward(self, x):
        return super(QBatchNorm2d, self).forward(self.act_quant(x))


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)

def quantize_grad(x, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
    return UniformQuantizeGrad().apply(x, num_bits, min_value, max_value, stochastic, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(
                input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(
                input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None,
                stochastic=False, inplace=False, enforce_true_zero=False, num_chunks=None, out_half=False):

        num_chunks = num_chunks = input.shape[
            0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            #min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
        if max_value is None:
            #max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = 0.
        qmax = 2.**num_bits - 1.
        #import pdb; pdb.set_trace()
        scale = (max_value - min_value) / (qmax - qmin)

        scale = max(scale, 1e-8)

        if enforce_true_zero:
            initial_zero_point = qmin - min_value / scale
            zero_point = 0.
            # make zero exactly represented
            if initial_zero_point < qmin:
                zero_point = qmin
            elif initial_zero_point > qmax:
                zero_point = qmax
            else:
                zero_point = initial_zero_point
            zero_point = int(zero_point)
            output.div_(scale).add_(zero_point)
        else:
            output.add_(-min_value).div_(scale).add_(qmin)

        if ctx.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        output.clamp_(qmin, qmax).round_()  # quantize
        if enforce_true_zero:
            output.add_(-zero_point).mul_(scale)  # dequantize
        else:
            output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.min_value is None:
            min_value = float(grad_output.min())
            # min_value = float(grad_output.view(
            # grad_output.size(0), -1).min(-1)[0].mean())
        else:
            min_value = ctx.min_value
        if ctx.max_value is None:
            max_value = float(grad_output.max())
            # max_value = float(grad_output.view(
            # grad_output.size(0), -1).max(-1)[0].mean())
        else:
            max_value = ctx.max_value
        grad_input = UniformQuantize().apply(grad_output, ctx.num_bits,
                                             min_value, max_value, ctx.stochastic, ctx.inplace)
        return grad_input, None, None, None, None, None


class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def extra_repr  (self):
        str_ = 'WA: {} - E: {}'.format(self.num_bits, self.num_bits_grad)
        return str_
    def forward(self, x):
        # x = self.quantize_input(x)
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()  # C x B x H x W
            if C * self.num_chunks * (B * H * W // self.num_chunks) == B * C * H * W:
                y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
            else:
                y = y.view(C, 1, B * H * W)
            mean_max = y.max(-1)[0].mean(-1)  # C
            mean_min = y.min(-1)[0].mean(-1)  # C
            mean = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)

            self.running_mean.detach().mul_(self.momentum).add_(
                mean * (1 - self.momentum))

            self.running_var.detach().mul_(self.momentum).add_(
                scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        # scale = quantize(scale, num_bits=self.num_bits, min_value=float(
        #     scale.min()), max_value=float(scale.max()))
        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)

        if self.weight is not None:
            # qweight = self.weight
            # qweight = quantize(self.weight, num_bits=self.num_bits,
            #                    min_value=float(self.weight.min()),
            #                    max_value=float(self.weight.max()))
            out = out * self.weight.view(1, self.weight.size(0), 1, 1)

        if self.bias is not None:
            # qbias = self.bias
            # qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + self.bias.view(1, self.bias.size(0), 1, 1)
        # if self.num_bits_grad is not None:
        #     if self.num_bits_grad != 32:
        #         out = quantize_grad(out, num_bits=self.num_bits_grad)

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


@with_incremental_state
class QMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        q_matmul_act=nn.Identity(),
        k_matmul_act=nn.Identity(),
        v_matmul_act=nn.Identity(),
        smax_matmul_act=nn.Identity(),
        k_q_matmul_error=nn.Identity(),
        smax_v_matmul_error=nn.Identity()
    ):
        super().__init__()

        self.q_matmul_act_quant = q_matmul_act
        self.k_matmul_act_quant = k_matmul_act
        self.v_matmul_act_quant = v_matmul_act
        self.smax_matmul_act_quant = smax_matmul_act
        self.k_q_matmul_err_quant = k_q_matmul_error
        self.smax_v_matmul_err_quant = smax_v_matmul_error


        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        # if (
        #     not self.onnx_trace
        #     and not is_tpu  # don't use PyTorch version on TPUs
        #     and incremental_state is None
        #     and not static_kv
        #     # A workaround for quantization to work. Otherwise JIT compilation
        #     # treats bias in linear module as method.
        #     and not torch.jit.is_scripting()
        # ):
        #     assert key is not None and value is not None
        #     return F.multi_head_attention_forward(
        #         query,
        #         key,
        #         value,
        #         self.embed_dim,
        #         self.num_heads,
        #         torch.empty([0]),
        #         torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        #         self.bias_k,
        #         self.bias_v,
        #         self.add_zero_attn,
        #         self.dropout_module.p,
        #         self.out_proj.weight,
        #         self.out_proj.bias,
        #         self.training or self.dropout_module.apply_during_inference,
        #         key_padding_mask,
        #         need_weights,
        #         attn_mask,
        #         use_separate_proj_weight=True,
        #         q_proj_weight=self.q_proj.weight,
        #         k_proj_weight=self.k_proj.weight,
        #         v_proj_weight=self.v_proj.weight,
        #     )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        # q *= self.scaling
        q = q*self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = QMultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = self.k_q_matmul_err_quant(
            torch.bmm(
                self.q_matmul_act_quant(q),
                self.k_matmul_act_quant(k.transpose(1, 2))
            )
        )
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = fairseq_utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = self.smax_v_matmul_err_quant(
            torch.bmm(
                self.smax_matmul_act_quant(attn_probs),
                self.v_matmul_act_quant(v))
        )
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value





