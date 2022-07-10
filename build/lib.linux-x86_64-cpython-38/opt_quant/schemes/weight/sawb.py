from opt_quant.schemes import QuantizeBase
from torch.autograd import Function
import torch


class sawbFn(Function):

    @staticmethod
    def forward(ctx, inp, scale, bits, representation, rounding_fn,
                kwargs):

        L1 = inp.abs().mean()
        L2 = inp.mul(inp).mean().sqrt()
        wclip = 0
        if bits == 1:
            wclip = L1
        elif bits == 2:
            wclip = -2.178 * L1 + 3.212 * L2
        elif bits == 4:
            wclip = -12.80 * L1 + 12.68 * L2
        elif bits == 5:
            wclip = -18.64 * L1 + 17.74 * L2

        wclip = torch.abs(wclip).item()
        clipped_weight = torch.clamp(inp, min=-wclip, max=wclip)

        # Normalize the data to [0, 1] range
        zero_one_val = ((clipped_weight / wclip) + 1) * 0.5

        # Quantize the [0, 1] range uniformly using bits
        quantized_zero_one = rounding_fn(
            scale * zero_one_val, bits=bits, **kwargs) / scale

        # renormalize to [-wclip, wclip]
        return wclip * (quantized_zero_one * 2 - 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


class sawb(QuantizeBase):

    def __init__(self,
                 representation='int',
                 rounding='nearest',
                 bitwidth=32,
                 **kwargs):
        super(sawb, self).__init__(representation=representation,
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
        return sawbFn.apply(inp, self.scale, self.bitwidth,
                            self.representation, self.quant_fun,
                            self.kwargs)