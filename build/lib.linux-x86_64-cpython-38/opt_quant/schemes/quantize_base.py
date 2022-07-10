import torch
from torch.quantization.observer import MovingAverageMinMaxObserver

import opt_quant.utils


class QuantizeBase(torch.nn.Module):
    r"""This is the base class for quantization schemes.
    """

    def __init__(self,
                 representation='fp',
                 rounding='nearest',
                 bitwidth=32,
                 **kwargs):
        super().__init__()
        self.representation = representation
        self.rounding = rounding
        self.bitwidth = bitwidth

        assert representation in ['int', 'fp', 'rdx2', 'rdx4'
                                  ], 'QuantizeBase: Invalid representation!'

        if representation == 'fp':
            if 'sig' in kwargs and 'man' in kwargs:
                self.sig_bits = kwargs['sig']
                self.man_bits = kwargs['man']
                if self.bitwidth == 32:
                    self.bitwidth = self.sig_bits + self.man_bits + 1
            elif bitwidth == 32:
                self.sig_bits = 8
                self.man_bits = 23
            else:
                raise ValueError('Invalid combinations of bits')
            assert self.bitwidth == self.sig_bits + self.man_bits + 1, 'Invalid combinations of bits'
            self.kwargs = {'sig': self.sig_bits, 'man': self.man_bits}
        else:
            self.kwargs = {}

        if representation in ['rdx2', 'rdx4']:
            assert self.bitwidth == 4, 'Only 4-bit quantization supports for Radix-r representation'

        self.quant_fun = getattr(
            opt_quant.utils, '{}_{}_quant'.format(self.representation,
                                                  self.rounding))