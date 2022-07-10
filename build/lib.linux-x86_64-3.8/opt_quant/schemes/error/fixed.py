from opt_quant.schemes import QuantizeBase
from torch.autograd import Function
import torch


class fixed_scaleFn(Function):

    @staticmethod
    def forward(ctx, inp, scale, bits, representation, rounding_fn,
                kwargs):

        ctx.constant = scale, rounding_fn, representation, bits, kwargs
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        scale, rounding_fn, representation, bits, kwargs = ctx.constant

        quantized_ = rounding_fn(
            scale * grad_output, bits=bits, **kwargs) / scale

        return quantized_, None, None, None, None, None


class fixed(QuantizeBase):

    def __init__(self,
                 representation='fp',
                 rounding='nearest',
                 bitwidth=32,
                 scale=1000.0,
                 **kwargs):
        super(fixed, self).__init__(representation=representation,
                                          rounding=rounding,
                                          bitwidth=bitwidth,
                                          **kwargs)
        self.scale = float(scale)

    def extra_repr(self) -> str:
        str_ = 'scale={}, '.format(self.scale)
        str_ += 'bits={}, ' \
                'rounding={}, representation={}, scale={}'.format(
            self.bitwidth, self.rounding, self.representation, self.scale)
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def forward(self, inp):
        if self.bitwidth == 32:
            return inp

        return fixed_scaleFn.apply(inp, self.scale, self.bitwidth,
                                   self.representation, self.quant_fun,
                                   self.kwargs)