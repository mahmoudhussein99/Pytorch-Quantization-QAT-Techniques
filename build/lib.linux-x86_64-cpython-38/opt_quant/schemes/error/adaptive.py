from opt_quant.schemes import QuantizeBase
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch


class adaptive_scaleFn(Function):

    @staticmethod
    def forward(ctx, inp, adaptive_scale, bits, representation, rounding_fn,
                kwargs):

        ctx.constant = adaptive_scale, rounding_fn, representation, bits, kwargs
        return inp

    @staticmethod
    def backward(ctx, grad_output):

        adaptive_scale, rounding_fn, representation, bits, kwargs = ctx.constant

        grad_max = torch.max(torch.abs(grad_output))
        adaptive_scale = torch.clamp(adaptive_scale, min=3.8147e-06, max=3.8147e+18)

        q_max = 2**(bits - 1) - 1
        if representation == 'fp':
            q_max = (2 - (0.5)**(kwargs['man'])) * 2**(
                2**(kwargs['sig'] - 1) - 1)
        elif representation == 'rdx2':
            q_max = 8
        elif representation == 'rdx4':
            q_max = 64
        if representation == 'int':
            q_max = 2**(bits - 1) - 1

        adaptive_scale_grad = torch.ones_like(adaptive_scale)
        if adaptive_scale * grad_max > q_max:
            adaptive_scale_grad *= -1
        elif adaptive_scale * grad_max < (q_max / 2):
            adaptive_scale_grad *= 1
        else:
            adaptive_scale_grad *= 0

        return rounding_fn(adaptive_scale * grad_output, bits=bits, **kwargs)/adaptive_scale, adaptive_scale_grad, None, None, None, None


class adaptive(QuantizeBase):

    def __init__(self,
                 representation='fp',
                 rounding='nearest',
                 bitwidth=32,
                 **kwargs):
        super(adaptive, self).__init__(representation=representation,
                                             rounding=rounding,
                                             bitwidth=bitwidth,
                                             **kwargs)
        self.adaptive_scale = Parameter(torch.Tensor([1000.0]))

    def extra_repr(self) -> str:
        str_ = 'scale={}, '.format(self.adaptive_scale[0])
        str_ +=  'bits={}, ' \
                       'rounding={}, representation={}'.format(
                       self.bitwidth, self.rounding, self.representation)
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def forward(self, inp):
        if self.bitwidth == 32:
            return inp

        return adaptive_scaleFn.apply(inp, self.adaptive_scale, self.bitwidth,
                                      self.representation, self.quant_fun,
                                      self.kwargs)