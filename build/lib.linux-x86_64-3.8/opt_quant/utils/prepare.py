import opt_quant
from copy import deepcopy
from torch.nn import *


# def resnet_replace_layers(args, model):
#     # Quantization layers
#     for n, module in model.named_children():
#         if 'layer' in n:
#             resnet_replace_one_qlayer(args, module, module_name=n+' ')

def replace_conv_layer(layer, args,
                       weight_qmodule,
                       act_qmodule,
                       error_qmodule,
                       grad_qmodule):

    for n, module in layer.named_children():


        if 'shortcut' in n and args.shortcut_quant == False: continue

        if len(list(module.children())) > 0:
            replace_conv_layer(module, args,
                               deepcopy(weight_qmodule),
                               deepcopy(act_qmodule),
                               deepcopy(error_qmodule),
                               deepcopy(grad_qmodule))

        if args.act_quantization:
            if isinstance(module, ReLU) or isinstance(module, ReLU6):
                act_qmodule.mode = 'unsigned'
                setattr(layer, n, Identity())
            else:
                act_qmodule.mode = 'signed'


        if isinstance(module, Conv2d):
            bias_status = False
            if module.bias is not None:
                bias_status = True

            quantized_conv = opt_quant.QConv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                               kernel_size=module.kernel_size, stride=module.stride,
                                               padding=module.padding, dilation=module.dilation, groups=module.groups,
                                               bias=bias_status,
                                               w_qmodule=deepcopy(weight_qmodule), act_qmodule=deepcopy(act_qmodule),
                                               err_qmodule=deepcopy(error_qmodule)
                                               )
            setattr(layer, n, quantized_conv)

def replace_linear_layer(layer, args,
                       weight_qmodule,
                       act_qmodule,
                       error_qmodule,
                       grad_qmodule):

    for n, module in layer.named_children():

        if len(list(module.children())) > 0:
            replace_linear_layer(module, args,
                               weight_qmodule,
                               act_qmodule,
                               error_qmodule,
                               grad_qmodule)

        if args.act_quantization:
            if isinstance(module, ReLU) or isinstance(module, ReLU6):
                act_qmodule.mode = 'unsigned'
                setattr(layer, n, Identity())
            else:
                act_qmodule.mode = 'signed'


        if isinstance(module, Linear):
            bias_status = False
            if module.bias is not None:
                bias_status = True

            quantized_linear = opt_quant.QLinear(in_features=module.in_features, out_features=module.out_features,
                                               bias=bias_status,
                                               w_qmodule=deepcopy(weight_qmodule), act_qmodule=deepcopy(act_qmodule),
                                               err_qmodule=deepcopy(error_qmodule)
                                               )
            setattr(layer, n, quantized_linear)




def model_prepare(model, args, layer_type='conv'):

    weight_qmodule = Identity()
    act_qmodule = Identity()
    error_qmodule = Identity()
    grad_qmodule = Identity()

    if args.weight_quantization:
        weight_qmodule = getattr(opt_quant.schemes.weight, args.weight_scheme)(representation=args.weight_rep,
                                                                               rounding=args.weight_round,
                                                                               bitwidth=args.weight_bits,
                                                                               **{'sig':args.weight_sig,
                                                                                  'man': args.weight_man})
    if args.act_quantization:
        act_qmodule = getattr(opt_quant.schemes.act, args.act_scheme)(representation=args.act_rep,
                                                                               rounding=args.act_round,
                                                                               bitwidth=args.act_bits,
                                                                               **{'sig':args.act_sig,
                                                                                  'man': args.act_man})

    if args.error_quantization:
        error_qmodule = getattr(opt_quant.schemes.error, args.error_scheme)(representation=args.error_rep,
                                                                               rounding=args.error_round,
                                                                               bitwidth=args.error_bits,
                                                                               **{'sig':args.error_sig,
                                                                                  'man': args.error_man,
                                                                                  'scale':args.error_scale})

    for n, module in model.named_children():
        if 'layer' in n and 'input' not in n:
            if layer_type == 'conv':
                replace_conv_layer(module, args, deepcopy(weight_qmodule), deepcopy(act_qmodule), deepcopy(error_qmodule), deepcopy(grad_qmodule))
            elif layer_type == 'linear':
                pass
            elif layer_type == 'mha':
                pass