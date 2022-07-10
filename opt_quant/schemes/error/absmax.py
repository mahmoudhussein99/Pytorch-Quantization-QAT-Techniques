from opt_quant.schemes import QuantizeBase
from torch.autograd import Function
import torch


class absmaxFn(Function):
    @staticmethod
    def forward(ctx, inp, bits, representation, rounding_fn,
                kwargs):
        ctx.constant = rounding_fn, representation, bits, kwargs
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        rounding_fn, representation, bits, kwargs = ctx.constant

        mxAbs = torch.max(torch.abs(grad_output))

        if  representation == 'rdx4':
            gradExp = torch.ceil(torch.log2(mxAbs) / 2)
            mxExp = 3
            curr_scale = 4 ** (gradExp - mxExp)
        elif representation == 'rdx2':
            gradExp = torch.ceil(torch.log2(mxAbs))
            mxExp = 3
            curr_scale = 2 ** (gradExp - mxExp)
        elif representation == 'fp':
            gradExp = torch.ceil(torch.log2(mxAbs))
            mxExp = 2 ** (kwargs['sig'] - 1) - 1
            curr_scale = 2 ** (gradExp - mxExp)

        scale = max(
            curr_scale,
            3.8147e-06)  # to make sure that we do not have overflow!
        # import pdb;
        # pdb.set_trace()
        quantized_ = rounding_fn(grad_output / scale, bits=bits, **kwargs)*scale

        return quantized_, None, None, None, None


class absmax(QuantizeBase):
    def __init__(self,
                 representation='fp',
                 rounding='nearest',
                 bitwidth=32,
                 **kwargs):

        assert representation in ['fp', 'rdx2', 'rdx4'], 'Only Floating-Point representations supported!'

        super(absmax, self).__init__(
            representation=representation,
            rounding=rounding,
            bitwidth=bitwidth,
            **kwargs
        )

    def extra_repr(self) -> str:
        str_ = ''
        str_ +=  'bits={}, ' \
                       'rounding={}, representation={}'.format(
                       self.bitwidth, self.rounding, self.representation)
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def forward(self, inp):
        if self.bitwidth == 32:
            return inp
        return absmaxFn.apply(inp, self.bitwidth,
                                   self.representation, self.quant_fun,
                                   self.kwargs)