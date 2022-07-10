//
// Created by Saleh on 4/5/21.
//
#include "floatx.hpp"
#include <torch/extension.h>

// CUDA forward
torch::Tensor makeTensor_cuda_main(
        torch::Tensor input,
        int64_t exp_bits, int64_t sig_bits);


float make_floatx_elem(float x, int64_t exp_bits, int64_t sig_bits){
    flx::floatxr<> fx_x(exp_bits, sig_bits);
    fx_x = x;
    return fx_x;
}


torch::Tensor makeTensor(torch::Tensor input, int64_t exp_bits, int64_t sig_bits) {
    auto output = torch::zeros_like(input);

    at::parallel_for(0, input.numel(), 0, [&](int64_t start, int64_t end){
        float* ptr = (float*)input.data_ptr();
        float* ptr_output = (float*)output.data_ptr();
        ptr = ptr + start;
        ptr_output = ptr_output + start;

        for (int i = start; i < end; ++i){
            *ptr_output = make_floatx_elem(*ptr, exp_bits, sig_bits);
            ptr++;ptr_output++;
        }
    });
    return output;

}

torch::Tensor makeTensor_cuda(torch::Tensor input, int64_t exp_bits, int64_t sig_bits) {
    return makeTensor_cuda_main(input, exp_bits, sig_bits);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


m.def("makeTensor", &makeTensor, "Convert Torch tensor to FloatX with arbitrary exp/sig bits",
    pybind11::arg("input"), pybind11::arg("exp_bits"),  pybind11::arg("sig_bits"));


m.def("makeTensor_cuda", &makeTensor_cuda, "Convert Torch tensor to FloatX with arbitrary exp/sig bits on GPU",
    pybind11::arg("input"), pybind11::arg("exp_bits"),  pybind11::arg("sig_bits"));
}

