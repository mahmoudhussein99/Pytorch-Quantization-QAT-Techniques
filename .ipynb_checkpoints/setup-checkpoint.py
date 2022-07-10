from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob
import sys

os_name = os.uname().sysname.lower()


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs

if os_name == 'linux':

    if 'cpu' in sys.argv:
        print('CPU SUPPORT!!')
        sys.argv.remove('cpu')

        setup(name='opt_quant',
              packages=find_packages(),
              ext_modules=[cpp_extension.CppExtension('floatxTensor', glob.glob('cpp/floatxTensor/*.cpp'), extra_compile_args=['-fopenmp'],

                                                      extra_link_args=['-lgomp'])],
              cmdclass={'build_ext': cpp_extension.BuildExtension},
              description='opt_quant: Optimal shift quantizer',
              version='0.0.2',
              python_requires='>=3.6')

    elif 'gpu' in sys.argv:
        print('GPU SUPPORT!')
        sys.argv.remove('gpu')

        setup(name='opt_quant',
              packages=find_packages(),
              # ext_modules=[cpp_extension.CppExtension('floatxTensor', glob.glob('floatxTensor/*.cpp'), extra_compile_args=['-fopenmp'],
              #                                         extra_link_args=['-lgomp'])],
              ext_modules=[
                  CUDAExtension('floatxTensor_gpu', [
                      'cpp/floatxTensor_gpu/main.cpp',
                      'cpp/floatxTensor_gpu/cuda_kernels.cu',
                  ]),
              ],
              cmdclass={'build_ext': cpp_extension.BuildExtension},
              description='opt_quant: Optimal shift quantizer',
              version='0.0.2',
              python_requires='>=3.6')
    else:
        raise RuntimeError("Specify cpu/gpu!")

else:
    setup(name='opt_quant',
          packages=find_packages(),
          ext_modules=[cpp_extension.CppExtension('floatxTensor',
                                                  glob.glob('cpp/floatxTensor/*.cpp'),
                                                  extra_compile_args=['-Xpreprocessor -fopenmp'],
                                                  extra_link_args=['-lomp'])],
          cmdclass={'build_ext': cpp_extension.BuildExtension},
          description='opt_quant: Optimal shift quantizer',
          version='0.0.2',
          python_requires='>=3.6'
          )

#
# setuptools.setup(
#     name="opt_quant",
#     version='0.0.1',
#     author="Saleh Ashkboos",
#     author_email="saleh.ashkboos@inf.ethz.ch",
#     description=("State of Quantization in Neural Network Training"),
#     python_requires='>=3.6',
#     packages=setuptools.find_packages(),
#     # install_requires=read_requirements()
# )