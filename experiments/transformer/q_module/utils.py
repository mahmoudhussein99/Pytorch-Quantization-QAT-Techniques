from experiments.utils import quant_parser, str2bool



def quant_args(parser):
    group = parser.add_argument_group("QuantizationArguments")

    # Quantization
    group.add_argument('--skip_shortcut', type=str2bool, default=False,
                        help='Whether we skip the shortcuts in RN for quantization or not (default: False)')


    group = quant_parser(group)

    return group

