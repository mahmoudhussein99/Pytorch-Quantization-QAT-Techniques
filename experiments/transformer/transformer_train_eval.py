from experiments.transformer.q_model import *
import argparse
import fairseq
from experiments.transformer.fairseq.fairseq import options
from experiments.transformer.fairseq.fairseq_cli.train import main as train_main
from experiments.utils import quant_parser

# from experiments.transformer.q_model import *


_original_save_checkpoint = fairseq.checkpoint_utils.save_checkpoint

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def quant_args(parser):
    group = parser.add_argument_group("QuantizationArguments")

    # Quantization
    group.add_argument('--skip_shortcut', type=str2bool, default=False,
                        help='Whether we skip the shortcuts in RN for quantization or not (default: False)')


    group = quant_parser(group)

    return group

def main_fun():
    parser = options.get_training_parser()
    quant_args(parser)
    args = options.parse_args_and_arch(parser)

    # TODO: Add error_bits to the args
    if args.error_rep != 'int':
        args.error_bits = args.error_sig + args.error_man + 1
    if args.last_error_rep != 'int':
        args.last_error_bits = args.last_error_sig + args.last_error_man + 1
    if args.mha_error_rep != 'int':
        args.mha_error_bits = args.mha_error_sig + args.mha_error_man + 1
    if args.fc_error_rep != 'int':
        args.fc_error_bits = args.fc_error_sig + args.fc_error_man + 1
    if args.mha_linear_error_rep != 'int':
        args.mha_linear_error_bits = args.mha_linear_error_sig + args.mha_linear_error_man + 1

    train_main(args)

if __name__ == "__main__":
    main_fun()