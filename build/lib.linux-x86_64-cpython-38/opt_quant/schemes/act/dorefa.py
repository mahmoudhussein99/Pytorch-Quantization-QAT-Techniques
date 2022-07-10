from opt_quant.schemes import QuantizeBase
from torch.autograd import Function
import torch


class DorefaQuantizer(Function):

    @staticmethod
    def forward(ctx, inp, mode, scale, bits, representation, rounding_fn,
                kwargs):

        return rounding_fn(inp * scale, bits=bits, **kwargs) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


class dorefa_act(QuantizeBase):

    def __init__(self,
                 representation='int',
                 rounding='nearest',
                 bitwidth=32,
                 mode='unsigned',
                 **kwargs):

        super(dorefa_act, self).__init__(representation=representation,
                                     rounding=rounding,
                                     bitwidth=bitwidth,
                                     **kwargs)
        assert mode in ['signed',
                        'unsigned'], 'DoreFa-A: mode could be signed/unsigned!'
        self.mode = mode
        self.kwargs = kwargs
        if mode == 'unsigned':
            assert self.representation == 'int', 'DoreFa-A: unsiged mode only supports int!'
            self.scale = 2**self.bitwidth - 1
        else:
            if self.representation == 'int':
                self.scale = 2**(self.bitwidth - 1) - 1
            else:
                if self.representation == 'fp':
                    self.scale = (2 - (0.5)**(kwargs['man'])) * 2**(
                        2**(kwargs['sig'] - 1) - 1)
                elif self.representation == 'rdx2':
                    self.scale = 8
                elif self.representation == 'rdx4':
                    self.scale = 64

    def extra_repr(self) -> str:
        str_ = '{}, '.format(self.mode.capitalize())
        str_ += 'bits={}, ' \
                'rounding={}, representation={}'.format(
            self.bitwidth, self.rounding, self.representation)
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def forward(self, inp):

        if self.bitwidth == 32:
            return inp

        if self.mode == 'signed':
            clipped_inp = torch.clamp(inp, -1, 1)
        else:
            clipped_inp = torch.clamp(inp, 0, 1)

        return DorefaQuantizer.apply(clipped_inp, self.mode, self.scale,
                                     self.bitwidth, self.representation,
                                     self.quant_fun, self.kwargs)
