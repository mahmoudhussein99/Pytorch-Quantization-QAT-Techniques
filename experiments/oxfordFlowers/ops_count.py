import pandas as pd
import argparse

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from q_model import ResNet18, add_op, MobileNetV2, RangeBN


def forward_hook(module, in_data, _):
    setattr(module, 'act', in_data[0])


def backward_hook(module, _, out_grad):
    setattr(module, 'err', out_grad[0])


def main():
    # prepare a model for CIFAR-10
    if args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'mobilenet':
        model = MobileNetV2()

    target_modules = (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.AvgPool2d, add_op)

    for module in model.modules():
        if isinstance(module, target_modules):
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    # prepare a tensor same shape as CIFAR-10 images
    bs = args.batch_size
    x = torch.randn(bs, 3, 32, 32)
    y = torch.tensor([0] * bs, dtype=torch.long)

    # call forward/backward
    loss = cross_entropy(model(x), y)
    loss.backward()

    # count numel & ops
    keys = ['to_quantize', 'weight', 'bias', 'act', 'err',
            'ops_w_a', 'ops_w_e', 'ops_a_e', 'op_w_o_g', 'ops_a_a',
            'rangeBN_ops_w_a', 'rangeBN_ops_w_e', 'rangeBN_ops_a_e', 'rangeBN_op_w_o_g', 'rangeBN_ops_a_a'
            ]
    counts = {key: [] for key in keys}
    names = []
    for name, module in model.named_modules():
        if isinstance(module, target_modules):
            names.append(name)

            if isinstance(module, nn.AvgPool2d) or isinstance(module, add_op):
                counts['weight'].append(0)
                counts['bias'].append(0)
            else:
                # register numel
                counts['weight'].append(module.weight.numel())
                if module.bias is not None:
                    counts['bias'].append(module.bias.numel())
                else:
                    counts['bias'].append(0)

            counts['act'].append(module.act.numel())
            counts['err'].append(module.err.numel())

            # register ops
            ops_w_a = ops_a_e = ops_w_e = op_w_o_g = ops_a_a  = 0
            rangeBN_ops_w_a = rangeBN_ops_a_e = rangeBN_ops_w_e = rangeBN_op_w_o_g = rangeBN_ops_a_a  = 0

            if isinstance(module, nn.Linear):
                f_out, f_in = module.weight.shape
                ops_w_a = ops_w_e = ops_a_e = bs * f_out * f_in
                op_w_o_g = module.weight.numel() *4  # weight upd
                counts['to_quantize'].append('No')
                rangeBN_ops_a_a = ops_a_a
                rangeBN_ops_w_a = ops_w_a
                rangeBN_ops_w_e = ops_w_e
                rangeBN_ops_a_e = ops_a_e
                rangeBN_op_w_o_g = op_w_o_g

            elif isinstance(module, nn.Conv2d):
                c_out, c_in, kh, kw = module.weight.shape
                _, _, h_in, w_in = module.act.shape
                _, _, h_out, w_out = module.err.shape
                ops_w_a = ops_a_e = bs * c_out * c_in * h_out * w_out * kh * kw
                ops_w_e = bs * c_in * c_out * h_in * h_in * kh * kw
                op_w_o_g = module.weight.numel() *4 # weight upd
                counts['to_quantize'].append('Yes')

                rangeBN_ops_a_a = ops_a_a
                rangeBN_ops_w_a = ops_w_a
                rangeBN_ops_w_e = ops_w_e
                rangeBN_ops_a_e = ops_a_e
                rangeBN_op_w_o_g = op_w_o_g


            elif isinstance(module, nn.BatchNorm2d):
                act_numel = module.act.numel()
                ops_a_a = act_numel * 6  # normalization  (var, mean, adding eps, sqrt, sub mean, div)
                ops_w_a = ops_w_e = act_numel * 2  # gamma, beta
                ops_a_e = act_numel * (5 + bs)  #from https://kevinzakka.github.io/2016/09/14/batch_normalization/
                op_w_o_g = module.weight.numel() * 2 * 4 # gamma, beta upd
                counts['to_quantize'].append('No')

                rangeBN_ops_a_a = act_numel * 3  # normalization  ( mean, sub mean, div)
                rangeBN_ops_w_a = rangeBN_ops_w_e = act_numel * 2  # gamma, beta
                rangeBN_ops_a_e = act_numel * 4  # from https://kevinzakka.github.io/2016/09/14/batch_normalization/
                rangeBN_op_w_o_g = module.weight.numel() * 2 * 4  # gamma, beta upd

            elif isinstance(module, add_op):
                act_numel = module.act.numel()
                ops_a_a = act_numel*2
                ops_w_a = 0
                ops_a_e = 0
                op_w_o_g = 0
                counts['to_quantize'].append('No')
                rangeBN_ops_a_a = ops_a_a
                rangeBN_ops_w_a = ops_w_a
                rangeBN_ops_w_e = ops_w_e
                rangeBN_ops_a_e = ops_a_e
                rangeBN_op_w_o_g = op_w_o_g

            elif isinstance(module, nn.AvgPool2d):
                act_numel = module.act.numel()
                err_numel = module.err.numel()
                ops_a_a = 4*bs*err_numel  # average over elements
                ops_w_a = 0
                ops_a_e = 4*bs*err_numel
                op_w_o_g = 0
                counts['to_quantize'].append('No')
                rangeBN_ops_a_a = ops_a_a
                rangeBN_ops_w_a = ops_w_a
                rangeBN_ops_w_e = ops_w_e
                rangeBN_ops_a_e = ops_a_e
                rangeBN_op_w_o_g = op_w_o_g

            if 'shortcut' in name:
                counts['to_quantize'][-1] = 'No'
                curr_name = names[-1]
                if curr_name[-1] == '0':
                    names[-1] = curr_name[:-2]+'.conv'
                else:
                    names[-1] = curr_name[:-2] + '.bn'

            counts['ops_w_a'].append(ops_w_a)
            counts['ops_a_e'].append(ops_a_e)
            counts['ops_w_e'].append(ops_w_e)
            counts['op_w_o_g'].append(op_w_o_g)
            counts['ops_a_a'].append(ops_a_a)

            counts['rangeBN_ops_w_a'].append(rangeBN_ops_w_a)
            counts['rangeBN_ops_a_e'].append(rangeBN_ops_a_e)
            counts['rangeBN_ops_w_e'].append(rangeBN_ops_w_e)
            counts['rangeBN_op_w_o_g'].append(rangeBN_op_w_o_g)
            counts['rangeBN_ops_a_a'].append(rangeBN_ops_a_a)



    df = pd.DataFrame(counts, index=names)
    print(df)
    df.to_csv(args.model+'.csv')
    df.to_pickle(args.model+args.pickle_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenet'])
    parser.add_argument('--pickle-path', type=str, default='_cifar10_counts.pickle')
    args = parser.parse_args()
    main()
