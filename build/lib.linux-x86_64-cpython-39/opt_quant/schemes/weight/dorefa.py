import torch
from opt_quant.schemes import QuantizeBase
from torch.autograd import Function


class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""

    @staticmethod
    def forward(ctx, inp):
        return torch.sign(inp) * torch.mean(torch.abs(inp))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DorefaQuantizer(Function):

    @staticmethod
    def forward(ctx, inp, scale, bits, representation, rounding_fn,
                kwargs):

        return rounding_fn(inp * scale, bits=bits, **kwargs) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


class dorefa_weight(QuantizeBase):

    def __init__(self,
                 representation='int',
                 rounding='nearest',
                 bitwidth=32,
                 **kwargs):

        super(dorefa_weight, self).__init__(representation=representation,
                                     rounding=rounding,
                                     bitwidth=bitwidth,
                                     **kwargs)
        self.kwargs = kwargs
        if self.representation == 'int':
            self.scale = 2**(self.bitwidth - 1) - 1
        elif self.representation == 'fp':
            self.scale = (2 - (0.5)**(kwargs['man'])) * 2**(
                2**(kwargs['sig'] - 1) - 1)
        elif self.representation == 'rdx2':
            self.scale = 8
        elif self.representation == 'rdx4':
            self.scale = 64

    def extra_repr(self) -> str:
        str_ = 'bits={}, ' \
                'rounding={}, representation={}'.format(
            self.bitwidth, self.rounding, self.representation)
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def forward(self, inp):
        if self.bitwidth == 1:
            w = ScaleSigner.apply(inp)
        elif self.bitwidth == 32:
            w = inp
        else:
            w = torch.tanh(inp)
            w = w / (2 * torch.max(torch.abs(w))) + 0.5
            w = 2 * DorefaQuantizer.apply(w, self.scale, self.bitwidth,
                                          self.representation, self.quant_fun,
                                          self.kwargs) - 1
        return w
