from opt_quant.schemes import QuantizeBase
from torch.nn.parameter import Parameter
import torch


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(inp, quant_fun, bits, **kwargs):
    y = quant_fun(inp, bits=bits, **kwargs)
    y_grad = inp
    return (y - y_grad).detach() + y_grad


class lsq_weight(QuantizeBase):

    def __init__(self,
                 representation='int',
                 rounding='nearest',
                 bitwidth=32,
                 per_channel=False,
                 **kwargs):
        super(lsq_weight, self).__init__(representation=representation,
                                  rounding=rounding,
                                  bitwidth=bitwidth,
                                  **kwargs)

        self.per_channel = per_channel
        self.s = Parameter(torch.tensor(float(1.0)))
        self.first_time_call = True
        self.kwargs = kwargs

        if self.representation == 'int':
            self.thd_neg = -2**(self.bitwidth - 1)
            self.thd_pos = 2**(self.bitwidth - 1) - 1
        else:
            if self.representation == 'fp':
                self.thd_pos = (2 - (0.5)**(kwargs['man'])) * 2**(
                    2**(kwargs['sig'] - 1) - 1)
            elif self.representation == 'rdx2':
                self.thd_pos = 8
            elif self.representation == 'rdx4':
                self.thd_pos = 64
            self.thd_neg = -self.thd_pos

    def extra_repr(self) -> str:
        str_ = 'bits={}, ' \
                'rounding={}, representation={}, s={} channel_wise={}'.format(
            self.bitwidth, self.rounding, self.representation, self.s, self.per_channel)
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def init_call(self, inp):
        if self.per_channel:
            self.s = Parameter(inp.detach().abs().mean(
                dim=list(range(1, inp.dim())), keepdim=True) * 2 /
                               (self.thd_pos**0.5))
        else:
            self.s = Parameter(inp.detach().abs().mean() * 2 /
                               (self.thd_pos**0.5))

    def forward(self, inp):
        if self.first_time_call:
            self.init_call(inp)
            self.first_time_call = False

        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * inp.numel())**0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * inp.numel())**0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        inp = inp / s_scale
        inp = torch.clamp(inp, self.thd_neg, self.thd_pos)
        inp = round_pass(inp, self.quant_fun, self.bitwidth,
                         **self.kwargs)
        inp = inp * s_scale
        return inp
