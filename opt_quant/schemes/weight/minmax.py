
from opt_quant.schemes import QuantizeBase
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch

class minmaxFn(Function):

    @classmethod
    def forward(cls, ctx, input, mode, bits, representation, quant_fn, min_value, max_value, kwargs):

        signs = torch.sign(input)
        qmin = 0.
        if mode == 'signed':
            input = torch.abs(input)
            qmax = (2. ** (bits-1)) - 1.
        else:
            qmax = (2. ** bits) - 1.


        # import pdb; pdb.set_trace()
        scale = (max_value - min_value) / (qmax - qmin)

        scale = max(scale, 1e-8)

        output = (input - min_value)/scale
        output += qmin

        dequantized = (quant_fn(torch.clamp(output, qmin, qmax), bits=bits, **kwargs) - qmin)

        return signs*((dequantized*scale)+min_value)


    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None


class minmax_weight(QuantizeBase):

    def __init__(self,
                 representation='int',
                 rounding='nearest',
                 bitwidth=32,
                 mode='signed',
                 per_channel=False,
                 momentum=0.1,
                 **kwargs):
        super(minmax_weight, self).__init__(representation=representation,
                                  rounding=rounding,
                                  bitwidth=bitwidth,
                                  **kwargs)
        assert mode == 'signed', 'Weights only quantized using signed mode!'
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.mode = mode
        self.kwargs = kwargs

    def extra_repr(self) -> str:
        str_ = '{}, '.format(self.mode)
        str_ += 'bits={}, ' \
                'rounding={}, representation={}'.format(
            self.bitwidth, self.rounding, self.representation)
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def forward(self, inp):
        if self.training:
            min_value = inp.detach().view(
                inp.size(0), -1).min(-1)[0].mean()
            max_value = inp.detach().view(
                inp.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max

        return minmaxFn.apply(inp, self.mode, self.bitwidth,
                        self.representation, self.quant_fun,
                        float(min_value), float(max_value),
                              self.kwargs)