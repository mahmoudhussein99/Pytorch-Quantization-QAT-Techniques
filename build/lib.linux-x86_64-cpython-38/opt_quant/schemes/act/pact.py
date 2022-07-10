from opt_quant.schemes import QuantizeBase
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch


class ReLU_PACTFn(Function):
    @staticmethod
    def forward(ctx, inp, mode, alpha, bits, representation, rounding_fn,
                kwargs):

        ctx.constant = mode
        ctx.save_for_backward(inp, alpha)

        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        y = torch.clamp(inp, min=0, max=alpha[0].item())
        scale = (2 ** bits - 1) / alpha
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        inp, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range
        lower_bound = inp < 0
        upper_bound = inp > alpha
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(inp, alpha).float()).view(-1)

        return dLdy_q * x_range.float(), None, grad_alpha, None, None, None, None


# This implementation of pact could be used with ReLU activation function in the network
# If the input is negative, we switch to signed pact (instead of clipping the input to 0).
# We only clip the values from the above!
class PACTFn(Function):

    @staticmethod
    def forward(ctx, inp, mode, alpha, bits, representation, rounding_fn,
                kwargs):

        ctx.constant = mode
        ctx.save_for_backward(inp, alpha)

        if mode == 'signed':
            y = torch.clamp(inp, min=-alpha[0].item(), max=alpha[0].item())
        else:
            y = torch.clamp(inp, max=alpha[0].item())

        if representation == 'int':
            if mode == 'signed':
                scale = (2 ** (bits - 1) - 1) / alpha
            else:
                scale = (2 ** bits - 1) / alpha
        elif representation == 'rdx2':
            scale = 8 / alpha
        elif representation == 'rdx4':
            scale = 64 / alpha
        elif representation == 'fp':
            scale = (2 - (0.5) ** (kwargs['man'])) * 2 ** (
                    2 ** (kwargs['sig'] - 1) - 1)

        return rounding_fn(y * scale, bits=bits, **kwargs) / scale

    @staticmethod
    def backward(ctx, dLdy_q):

        # We get dL / dy_q as a gradient
        inp, alpha, = ctx.saved_tensors
        mode = ctx.constant

        grad_alpha = dLdy_q.clone()
        grad_input = dLdy_q.clone()

        if mode == 'unsigned':
            grad_alpha[inp < alpha] = 0
            grad_input[inp > alpha] = 0
        else:
            grad_alpha[(inp < alpha) & (inp > -alpha)] = 0
            grad_input[inp > alpha] = 0
            grad_input[inp < -alpha] = 0

        return grad_input, None, torch.sum(grad_alpha).view(-1), None, None, None, None

class pact(QuantizeBase):

    def __init__(self,
                 representation='int',
                 rounding='nearest',
                 bitwidth=32,
                 mode='unsigned',
                 enable_relu=False,
                 **kwargs):
        super(pact, self).__init__(representation=representation,
                                   rounding=rounding,
                                   bitwidth=bitwidth,
                                   **kwargs)

        assert mode in ['signed',
                        'unsigned'], 'PACT: mode could be signed/unsigned!'
        self.mode = mode
        alpha = 6.0
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']

        self.enable_relu = enable_relu
        self.alpha = Parameter(torch.tensor([float(alpha)]))
        self.kwargs = kwargs

    def extra_repr(self) -> str:
        str_ = '{}, '.format(self.mode.capitalize())
        str_ += 'ReLU-Act={}, bits={}, ' \
                'rounding={}, representation={}, alpha={}'.format(
            self.enable_relu, self.bitwidth, self.rounding, self.representation, self.alpha[0])
        if self.representation == 'fp':
            str_ += '(sig={}, man={})'.format(self.sig_bits, self.man_bits)

        return str_

    def forward(self, inp):
        if self.bitwidth == 32:
            return inp

        if self.mode == 'unsigned':
            return ReLU_PACTFn.apply(inp, self.mode, torch.abs(self.alpha), self.bitwidth,
                                self.representation, self.quant_fun,
                                self.kwargs)

        return PACTFn.apply(inp, self.mode, torch.abs(self.alpha), self.bitwidth,
                            self.representation, self.quant_fun,
                            self.kwargs)
