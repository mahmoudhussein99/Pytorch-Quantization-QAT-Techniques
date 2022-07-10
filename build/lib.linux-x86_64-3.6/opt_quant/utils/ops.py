import torch
# import opt_quant.ieee_quant
if torch.cuda.is_available():
    from floatxTensor_gpu import makeTensor_cuda as ieee_quant
else:
    from floatxTensor import makeTensor as ieee_quant


def int_nearest_quant(inp: torch.Tensor, bits=32, **bitwidth_kwargs):
    "Round to INT-bit with Nearest Rounding"

    # saturation
    if torch.min(inp) < 0:
        q_n = -2**(bits - 1)
        q_p = 2**(bits - 1) - 1
    else:
        q_n = 0
        q_p = 2**bits - 1

    inp.clamp_(q_n, q_p)
    return torch.round(inp)


def int_stochastic_quant(inp: torch.Tensor, bits=32, **bitwidth_kwargs):
    "Round to INT-bit with Stochastic Rounding"

    if torch.min(inp) < 0:
        q_n = -2**(bits - 1)
        q_p = 2**(bits - 1) - 1
    else:
        q_n = 0
        q_p = 2**bits - 1

    inp.clamp_(q_n, q_p)

    signs_ = torch.sign(inp)
    unsigned_inp = torch.abs(inp)

    v = torch.floor(unsigned_inp + torch.rand_like(inp))

    return v * signs_


def fp_nearest_quant(inp: torch.Tensor, bits=32, **bitwidth_kwargs):
    "Quantize using IEEE representatoin"

    e = bitwidth_kwargs['sig']
    m = bitwidth_kwargs['man']

    max_representable = (2 - (0.5) ** (m)) * 2 ** (2 ** (e - 1) - 1)

    sigs = torch.sign(inp)
    unsigned_inp = torch.abs(inp)

    return ieee_quant(unsigned_inp.clamp_(0, max_representable), e, m)*sigs

def fp_stochastic_quant(inp: torch.Tensor, bits=32, **bitwidth_kwargs):
    pass


def rdx2_nearest_quant(inp: torch.Tensor, bits=4, **bitwidth_kwargs):
    assert bits == 4, 'Only 4-bit quantization supports ' \
                      'for Radix-r representation'

    signs = torch.sign(inp)
    inp = torch.abs(inp * (4 / 3))
    ebit = torch.floor(torch.log2(inp))
    underflow_ = ebit < -3
    overflow_ = ebit >= 3
    output = 2**ebit
    output[underflow_] = 0
    output[overflow_] = 8
    return output * signs


def rdx2_stochastic_quant(inp: torch.Tensor, bits=4, **bitwidth_kwargs):
    assert bits == 4, 'Only 4-bit quantization supports ' \
                      'for Radix-r representation'

    signs = torch.sign(inp)
    unsigned_inp = torch.abs(inp)

    quantized = rdx2_nearest_quant(unsigned_inp)

    q_max = 2**3
    q_min = 2**-3

    underflow = quantized == 0
    overflow = unsigned_inp > q_max

    quantized_sign = torch.sign(unsigned_inp - quantized)

    prob = torch.abs(1 - (unsigned_inp / quantized))

    prob[(overflow | underflow)] = 0.0
    prob[quantized_sign < 0] *= 2

    to_change = torch.bernoulli(prob)
    quantized *= torch.pow(2, quantized_sign * to_change)
    quantized[
        quantized < q_min] = 0  # remove the fake quantization point in c_0/2

    quantized[underflow] += torch.bernoulli(
        unsigned_inp[underflow] / q_min) * q_min

    return quantized.clamp_(0, 8) * signs


def rdx4_nearest_quant(inp: torch.Tensor, bits=4, **bitwidth_kwargs):
    assert bits == 4, 'Only 4-bit quantization supports ' \
                      'for Radix-r representation'

    signs = torch.sign(inp)
    inp = torch.abs(inp * 1.6)
    ebit = torch.floor(torch.log2(inp) / 2)
    underflow_ = ebit < -3
    overflow_ = ebit >= 3
    output = 4**ebit
    output[underflow_] = 0
    output[overflow_] = 64
    return output * signs


def rdx4_stochastic_quant(inp: torch.Tensor, bits=4, **bitwidth_kwargs):
    assert bits == 4, 'Only 4-bit quantization supports ' \
                      'for Radix-r representation'

    signs = torch.sign(inp)
    unsigned_inp = torch.abs(inp)

    quantized = rdx4_nearest_quant(unsigned_inp)

    q_max = 4**3
    q_min = 4**-3

    underflow = quantized == 0
    overflow = unsigned_inp > q_max

    quantized_sign = torch.sign(unsigned_inp - quantized)

    prob = 1 / 3 * torch.abs(1 - (unsigned_inp / quantized))
    prob[(overflow | underflow)] = 0

    prob[quantized_sign < 0] *= 4

    to_change = torch.bernoulli(prob)
    quantized *= torch.pow(4, quantized_sign * to_change)
    quantized[
        quantized < q_min] = 0  # remove the fake quantization point in c_0/2

    quantized[underflow] += torch.bernoulli(
        unsigned_inp[underflow] / q_min) * q_min

    return quantized * signs
