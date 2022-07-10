#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "floatx.hpp"

#include <vector>


float
__device__  make_floatx_elem_cuda(float x, int64_t exp_bits, int64_t sig_bits){
    flx::floatxr<> fx_x(exp_bits, sig_bits);
    fx_x = x;
    return fx_x;
}

__global__ void makeTensor_cuda_kernel(
        const  float*  __restrict__ input,
        float* __restrict__ output,
        size_t tensor_size,
        int64_t exp_bits, int64_t sig_bits) {
    const int  idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < tensor_size)
        output[idx] = make_floatx_elem_cuda(input[idx], exp_bits, sig_bits);
}

torch::Tensor makeTensor_cuda_main(
    torch::Tensor input,
    int64_t exp_bits, int64_t sig_bits){

    const int tensor_size = input.numel();
    const int threads = 1024;

    auto output = torch::zeros_like(input);

    makeTensor_cuda_kernel<<<(tensor_size/threads) + 1, threads>>>(input.data<float>(),
                                                                   output.data<float>(),
                                                                   tensor_size,
                                                                   exp_bits,
                                                                   sig_bits);

    return output;
}
