import argparse
import sys
import time

import torch
import torch.nn as nn
import opt_quant


def get_bn_modules(args):
    bn_w_qmodule = nn.Identity()
    bn_act_qmodule = nn.Identity()
    bn_err_qmodule = nn.Identity()

    # Weights
    if args.bn_weight_bits != 32:
        bn_w_qmodule = getattr(opt_quant, args.bn_weight_qmode)(
            representation=args.bn_weight_rep,
            rounding=args.bn_weight_rounding,
            bitwidth=args.bn_weight_bits,
            man=args.bn_weight_man, sig=args.bn_weight_sig
        )

    # Activations
    if args.bn_act_bits != 32:
        bn_act_qmodule = getattr(opt_quant, args.bn_act_qmode)(
            representation=args.bn_act_rep,
            rounding=args.bn_act_rounding,
            bitwidth=args.bn_act_bits,
            mode='signed', #All BN have signed input in RN network!
            man=args.bn_act_man, sig=args.bn_act_sig
        )

    # Error
    if args.bn_error_bits != 32:
        bn_err_qmodule = getattr(opt_quant, args.bn_error_qmode)(
                        representation=args.bn_error_rep,
                        rounding=args.bn_error_rounding,
                        bitwidth=args.bn_error_bits,
                        scale=args.bn_error_scale,
                        sig=args.bn_error_sig,
                        man=args.bn_error_man,
                    )

    return bn_w_qmodule, bn_act_qmodule, bn_err_qmodule


def get_ln_modules(args):
    ln_act_qmodule = nn.Identity()

    # Activations
    if args.ln_act_bits != 32:
        ln_act_qmodule = getattr(opt_quant, args.ln_act_qmode)(
            representation=args.ln_act_rep,
            rounding=args.ln_act_rounding,
            bitwidth=args.ln_act_bits,
            mode='signed',
            man=args.ln_act_man, sig=args.ln_act_sig
        )

    return ln_act_qmodule


def get_first_layer_modules(args):
    first_w_qmodule = nn.Identity()
    first_act_qmodule = nn.Identity()
    first_err_qmodule = nn.Identity()

    # Weights
    if args.first_weight_bits != 32:
        first_w_qmodule = getattr(opt_quant, args.first_weight_qmode)(
            representation=args.first_weight_rep,
            rounding=args.first_weight_rounding,
            bitwidth=args.first_weight_bits,
            man=args.first_weight_man, sig=args.first_weight_sig
        )

    # Activations
    if args.first_act_bits != 32:
        first_act_qmodule = getattr(opt_quant, args.first_act_qmode)(
            representation=args.first_act_rep,
            rounding=args.first_act_rounding,
            bitwidth=args.first_act_bits,
            mode='signed', #First Layer has signed input in RN network!
            man=args.first_act_man, sig=args.first_act_sig
        )

    # Error
    if args.first_error_bits != 32:
        first_err_qmodule = getattr(opt_quant, args.first_error_qmode)(
            representation=args.first_error_rep,
            rounding=args.first_error_rounding,
            bitwidth=args.first_error_bits,
            scale=args.first_error_scale,
            sig=args.first_error_sig,
            man=args.first_error_man,
        )

    return first_w_qmodule, first_act_qmodule, first_err_qmodule


def get_mha_modules(args):
    mha_act_qmodule = nn.Identity()
    mha_err_qmodule = nn.Identity()

    # Activations
    if args.mha_act_bits != 32:
        mha_act_qmodule = getattr(opt_quant, args.mha_act_qmode)(
            representation=args.mha_act_rep,
            rounding=args.mha_act_rounding,
            bitwidth=args.mha_act_bits,
            mode='signed',
            man=args.mha_act_man, sig=args.mha_act_sig
        )

    # Error
    if args.mha_error_bits != 32:
        mha_err_qmodule = getattr(opt_quant, args.mha_error_qmode)(
            representation=args.mha_error_rep,
            rounding=args.mha_error_rounding,
            bitwidth=args.mha_error_bits,
            scale=args.mha_error_scale,
            sig=args.mha_error_sig,
            man=args.mha_error_man,
        )

    return mha_act_qmodule, mha_err_qmodule


def get_last_layer_modules(args):
    last_w_qmodule = nn.Identity()
    last_act_qmodule = nn.Identity()
    last_err_qmodule = nn.Identity()

    # Weights
    if args.last_weight_bits != 32:
        last_w_qmodule = getattr(opt_quant, args.last_weight_qmode)(
            representation=args.last_weight_rep,
            rounding=args.last_weight_rounding,
            bitwidth=args.last_weight_bits,
            man=args.last_weight_man, sig=args.last_weight_sig
        )

    # Activations
    if args.last_act_bits != 32:
        last_act_qmodule = getattr(opt_quant, args.last_act_qmode)(
            representation=args.last_act_rep,
            rounding=args.last_act_rounding,
            bitwidth=args.last_act_bits,
            mode='signed',
            man=args.last_act_man, sig=args.last_act_sig
        )

    # Error
    if args.last_error_bits != 32:
        last_err_qmodule = getattr(opt_quant, args.last_error_qmode)(
            representation=args.last_error_rep,
            rounding=args.last_error_rounding,
            bitwidth=args.last_error_bits,
            scale=args.last_error_scale,
            sig=args.last_error_sig,
            man=args.last_error_man,
        )

    return last_w_qmodule, last_act_qmodule, last_err_qmodule


def get_q_modules(args):
    w_qmodule = nn.Identity()
    act_qmodule = nn.Identity()
    err_qmodule = nn.Identity()

    # Weights
    if args.weight_bits != 32:
        w_qmodule = getattr(opt_quant, args.weight_qmode)(
            representation=args.weight_rep,
            rounding=args.weight_rounding,
            bitwidth=args.weight_bits,
            man=args.weight_man, sig=args.weight_sig
        )

    # Activations
    if args.act_bits != 32:
        act_qmodule = getattr(opt_quant, args.act_qmode)(
            representation=args.act_rep,
            rounding=args.act_rounding,
            bitwidth=args.act_bits,
            mode='unsigned', #All CONV layers have unsgined input in RN networks!
            man=args.act_man, sig=args.act_sig
        )

    # Error
    if args.error_bits != 32:
        err_qmodule = getattr(opt_quant, args.error_qmode)(
                        representation=args.error_rep,
                        rounding=args.error_rounding,
                        bitwidth=args.error_bits,
                        scale=args.error_scale,
                        sig=args.error_sig,
                        man=args.error_man,
                    )

    return w_qmodule, act_qmodule, err_qmodule


def get_mha_linear_modules(args):
    mha_linear_w_qmodule = nn.Identity()
    mha_linear_act_qmodule = nn.Identity()
    mha_linear_err_qmodule = nn.Identity()

    # Weights
    if args.mha_linear_weight_bits != 32:
        mha_linear_w_qmodule = getattr(opt_quant, args.mha_linear_weight_qmode)(
            representation=args.mha_linear_weight_rep,
            rounding=args.mha_linear_weight_rounding,
            bitwidth=args.mha_linear_weight_bits,
            man=args.mha_linear_weight_man, sig=args.mha_linear_weight_sig
        )

    # Activations
    if args.mha_linear_act_bits != 32:
        mha_linear_act_qmodule = getattr(opt_quant, args.mha_linear_act_qmode)(
            representation=args.mha_linear_act_rep,
            rounding=args.mha_linear_act_rounding,
            bitwidth=args.mha_linear_act_bits,
            mode='signed',
            man=args.mha_linear_act_man, sig=args.mha_linear_act_sig
        )

    # Error
    if args.mha_linear_error_bits != 32:
        mha_linear_err_qmodule = getattr(opt_quant, args.mha_linear_error_qmode)(
                        representation=args.mha_linear_error_rep,
                        rounding=args.mha_linear_error_rounding,
                        bitwidth=args.mha_linear_error_bits,
                        scale=args.mha_linear_error_scale,
                        sig=args.mha_linear_error_sig,
                        man=args.mha_linear_error_man,
                    )

    return mha_linear_w_qmodule, mha_linear_act_qmodule, mha_linear_err_qmodule


def get_fc_q_modules(args):
    fc_w_qmodule = nn.Identity()
    fc_act_qmodule = nn.Identity()
    fc_err_qmodule = nn.Identity()

    # Weights
    if args.fc_weight_bits != 32:
        fc_w_qmodule = getattr(opt_quant, args.fc_weight_qmode)(
            representation=args.fc_weight_rep,
            rounding=args.fc_weight_rounding,
            bitwidth=args.fc_weight_bits,
            man=args.fc_weight_man, sig=args.fc_weight_sig
        )

    # Activations
    if args.fc_act_bits != 32:
        fc_act_qmodule = getattr(opt_quant, args.fc_act_qmode)(
            representation=args.fc_act_rep,
            rounding=args.fc_act_rounding,
            bitwidth=args.fc_act_bits,
            mode=args.fc_act_mode,
            man=args.fc_act_man, sig=args.fc_act_sig
        )

    # Error
    if args.fc_error_bits != 32:
        fc_err_qmodule = getattr(opt_quant, args.fc_error_qmode)(
                        representation=args.fc_error_rep,
                        rounding=args.fc_error_rounding,
                        bitwidth=args.fc_error_bits,
                        scale=args.fc_error_scale,
                        sig=args.fc_error_sig,
                        man=args.fc_error_man,
                    )

    return fc_w_qmodule, fc_act_qmodule, fc_err_qmodule


def get_sgd_optimizer(net, args):
    if args.optim_qmode == 'bnb':

        if torch.cuda.is_available():
            import bitsandbytes as bnb
            print('BnB 8-bit Optimizer.....')
            return bnb.optim.SGD(net.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay,
                                 optim_bits=args.optim_bits)
        else:
            raise Exception('bnb only works when cuda is available')

    return torch.optim.SGD(net.parameters(), lr=args.lr,
                           momentum=args.momentum, weight_decay=args.weight_decay)


TERM_WIDTH= 80
TOTAL_BAR_LENGTH = 65.


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds -= days * 3600 * 24
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    secondsf = int(seconds)
    seconds -= secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(TERM_WIDTH-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(TERM_WIDTH-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tuple_float_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)


def quant_parser(parser):
    # ---------General---------

    # Weight Quantization:
    parser.add_argument('--weight_qmode', type=str, default='none',
                     choices=['lsq_weight', 'sawb', 'dorefa_weight', 'minmax_weight','none'],
                     help='The quantization mode for weights!')
    parser.add_argument('--weight_rep',
                     type=str,
                     default='int',
                     choices=['int', 'fp', 'rdx2', 'rdx4'],
                     help='Weight representation (default: int)')
    parser.add_argument('--weight_bits', type=int, default=32, help='Weight bits(default:32)')
    parser.add_argument('--weight_rounding',
                     type=str,
                     default='nearest',
                     choices=['nearest', 'stochastic'],
                     help='Weight rounding method (default: nearest)')

    parser.add_argument('--weight_per_channel', type=bool, default=False, help='Weight lsq per_channel(default:False)')
    parser.add_argument('--weight_sig', type=int, default=23, help='weight Mantissa (default:23)')
    parser.add_argument('--weight_man', type=int, default=8, help='weight Significant (default:8)')

    # Activation Quantization:
    parser.add_argument('--act_qmode', type=str, default='none',
                     choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act','none'],
                     help='The quantization mode for activations!')
    parser.add_argument('--act_rep',
                     type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                     help='Act representation (default: int)')

    parser.add_argument('--act_bits', type=int, default=32, help='Act bits(default:32)')

    parser.add_argument('--act_rounding',
                     type=str, default='nearest', choices=['nearest', 'stochastic'],
                     help='Act rounding method (default: nearest)')

    parser.add_argument('--act_mode', type=str, default='signed',
                     choices=['unsigned', 'signed'],
                     help='Act dorefa mode(default:signed)')

    parser.add_argument('--act_per_channel', type=bool, default=False, help='Act lsq per_channel(default:False)')
    parser.add_argument('--act_sig', type=int, default=23, help='Act Mantissa (default:23)')
    parser.add_argument('--act_man', type=int, default=8, help='Act Significant (default:8)')

    # Act specific Aurguments
    parser.add_argument('--pact_reg', type=float, default=0.0001)


    # Error:
    parser.add_argument('--error_qmode', type=str, default='none',
                     choices=['adaptive', 'fixed', 'absmax', 'none'],
                     help='The quantization mode for errors!')
    parser.add_argument('--error_rep', type=str, default='fp', choices=['int', 'fp', 'rdx2', 'rdx4'],
                     help='Error representation (default: int)')

    parser.add_argument('--error_rounding',
                     type=str, default='nearest', choices=['nearest', 'stochastic'],
                     help='Error rounding method (default: nearest)')

    parser.add_argument('--error_sig', type=int, default=8, help='error Significant (default:8)')
    parser.add_argument('--error_man', type=int, default=23, help='error Mantissa (default:23)')
    parser.add_argument('--error_scale', type=float, default=100000.0, help='Error scale (default: 100000)')


    #---------BN---------

    #    Activation:
    parser.add_argument('--bn', type=str, default='BN', choices=['BN', 'RangeBN'],
                        help='Batch Norm Type')

    parser.add_argument('--bn_act_qmode', type=str, default='none',
                        choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act', 'none'],
                        help='The quantization mode for batch norm activations!')
    parser.add_argument('--bn_act_rep',
                        type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Act representation of batch norm(default: int)')

    parser.add_argument('--bn_act_bits', type=int, default=32, help='Batch norm Act bits(default:32)')

    parser.add_argument('--bn_act_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='Batch Norm Act rounding method (default: nearest)')

    parser.add_argument('--bn_act_per_channel', type=bool, default=False, help='Batch Norm Act lsq per_channel(default:False)')
    parser.add_argument('--bn_act_sig', type=int, default=23, help='Batch Norm Act Mantissa (default:23)')
    parser.add_argument('--bn_act_man', type=int, default=8, help='Batch Norm Act Significant (default:8)')


    #   Weights:
    parser.add_argument('--bn_weight_qmode', type=str, default='none',
                        choices=['lsq_weight', 'sawb', 'dorefa_weight', 'minmax_weight', 'none'],
                        help='The quantization mode for batch norm weights!')
    parser.add_argument('--bn_weight_rep',
                        type=str,
                        default='int',
                        choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Weight representation for batch norm(default: int)')
    parser.add_argument('--bn_weight_bits', type=int, default=32, help='Weight bits for batch norm (default:32)')
    parser.add_argument('--bn_weight_rounding',
                        type=str,
                        default='nearest',
                        choices=['nearest', 'stochastic'],
                        help='Weight rounding method for batchnorm (default: nearest)')

    parser.add_argument('--bn_weight_per_channel', type=bool, default=False, help='Weight lsq per_channel for bn(default:False)')
    parser.add_argument('--bn_weight_sig', type=int, default=23, help='weight Mantissa for bn(default:23)')
    parser.add_argument('--bn_weight_man', type=int, default=8, help='weight Significant for bn(default:8)')



    #   Error:
    parser.add_argument('--bn_error_qmode', type=str, default='none',
                        choices=['adaptive', 'fixed', 'absmax', 'none'],
                        help='The quantization mode for bn errors!')
    parser.add_argument('--bn_error_rep', type=str, default='fp', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='BN Error representation (default: int)')

    parser.add_argument('--bn_error_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='BN Error rounding method (default: nearest)')

    parser.add_argument('--bn_error_sig', type=int, default=8, help='BN error Significant (default:8)')
    parser.add_argument('--bn_error_man', type=int, default=23, help='BN error Mantissa (default:23)')
    parser.add_argument('--bn_error_scale', type=float, default=100000.0, help='BN Error scale (default: 100000)')


    # ---------First Layer---------

    #     Act
    parser.add_argument('--first_act_qmode', type=str, default='none',
                        choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act', 'none'],
                        help='The quantization mode for First Layer activations!')
    parser.add_argument('--first_act_rep',
                        type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Act representation of First Layer(default: int)')

    parser.add_argument('--first_act_bits', type=int, default=32, help='First Layer Act bits(default:32)')

    parser.add_argument('--first_act_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='First Layer Act rounding method (default: nearest)')

    parser.add_argument('--first_act_per_channel', type=bool, default=False, help='First Layer Act lsq per_channel(default:False)')
    parser.add_argument('--first_act_sig', type=int, default=23, help='First Layer Act Mantissa (default:23)')
    parser.add_argument('--first_act_man', type=int, default=8, help='First Layer Act Significant (default:8)')



    #   Weights:
    parser.add_argument('--first_weight_qmode', type=str, default='none',
                        choices=['lsq_weight', 'sawb', 'dorefa_weight', 'minmax_weight', 'none'],
                        help='The quantization mode for First Layer weights!')
    parser.add_argument('--first_weight_rep',
                        type=str,
                        default='int',
                        choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Weight representation for First Layer(default: int)')
    parser.add_argument('--first_weight_bits', type=int, default=32, help='Weight bits for First Layer (default:32)')
    parser.add_argument('--first_weight_rounding',
                        type=str,
                        default='nearest',
                        choices=['nearest', 'stochastic'],
                        help='Weight rounding method for First Layer (default: nearest)')

    parser.add_argument('--first_weight_per_channel', type=bool, default=False, help='Weight lsq per_channel for First Layer(default:False)')
    parser.add_argument('--first_weight_sig', type=int, default=23, help='weight Mantissa for First Layer(default:23)')
    parser.add_argument('--first_weight_man', type=int, default=8, help='weight Significant for First Layer(default:8)')



    #   Error:
    parser.add_argument('--first_error_qmode', type=str, default='none',
                        choices=['adaptive', 'fixed', 'absmax', 'none'],
                        help='The quantization mode for First Layer errors!')
    parser.add_argument('--first_error_rep', type=str, default='fp', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='First Layer Error representation (default: int)')

    parser.add_argument('--first_error_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='First Layer Error rounding method (default: nearest)')

    parser.add_argument('--first_error_sig', type=int, default=8, help='First Layer error Significant (default:8)')
    parser.add_argument('--first_error_man', type=int, default=23, help='First Layer error Mantissa (default:23)')
    parser.add_argument('--first_error_scale', type=float, default=1000000.0, help='First Layer Error scale (default: 1000000.0)')


    # ---------FC---------

    # Weight
    parser.add_argument('--fc_weight_qmode', type=str, default='none',
                        choices=['lsq_weight', 'sawb', 'dorefa_weight', 'minmax_weight','none'],
                        help='The quantization mode for weights!')
    parser.add_argument('--fc_weight_rep',
                        type=str,
                        default='int',
                        choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Weight representation (default: int)')
    parser.add_argument('--fc_weight_bits', type=int, default=32, help='Weight bits(default:32)')
    parser.add_argument('--fc_weight_rounding',
                        type=str,
                        default='nearest',
                        choices=['nearest', 'stochastic'],
                        help='Weight rounding method (default: nearest)')

    parser.add_argument('--fc_weight_per_channel', type=bool, default=False, help='Weight lsq per_channel(default:False)')
    parser.add_argument('--fc_weight_sig', type=int, default=23, help='weight Mantissa (default:23)')
    parser.add_argument('--fc_weight_man', type=int, default=8, help='weight Significant (default:8)')


    # Activation
    parser.add_argument('--fc_act_qmode', type=str, default='none',
                        choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act','none'],
                        help='The quantization mode for activations!')
    parser.add_argument('--fc_act_rep',
                        type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Act representation (default: int)')

    parser.add_argument('--fc_act_bits', type=int, default=32, help='Act bits(default:32)')

    parser.add_argument('--fc_act_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='Act rounding method (default: nearest)')

    parser.add_argument('--fc_act_mode', type=str, default='signed',
                        choices=['unsigned', 'signed'],
                        help='Act dorefa mode(default:signed)')

    parser.add_argument('--fc_act_per_channel', type=bool, default=False, help='Act lsq per_channel(default:False)')
    parser.add_argument('--fc_act_sig', type=int, default=23, help='Act Mantissa (default:23)')
    parser.add_argument('--fc_act_man', type=int, default=8, help='Act Significant (default:8)')

    # Act specific Aurguments
    parser.add_argument('--fc_pact_reg', type=float, default=0.0001)


    # Error:
    parser.add_argument('--fc_error_qmode', type=str, default='none',
                        choices=['adaptive', 'fixed', 'absmax', 'none'],
                        help='The quantization mode for errors!')
    parser.add_argument('--fc_error_rep', type=str, default='fp', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Error representation (default: int)')

    parser.add_argument('--fc_error_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='Error rounding method (default: nearest)')

    parser.add_argument('--fc_error_sig', type=int, default=8, help='error Significant (default:8)')
    parser.add_argument('--fc_error_man', type=int, default=23, help='error Mantissa (default:23)')
    parser.add_argument('--fc_error_scale', type=float, default=100000.0, help='Error scale (default: 100000)')



    # ---------MHA---------

    #     Act
    parser.add_argument('--mha_act_qmode', type=str, default='none',
                        choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act', 'none'],
                        help='The quantization mode for MHA Layers activations!')
    parser.add_argument('--mha_act_rep',
                        type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Act representation of MHA Layer(default: int)')

    parser.add_argument('--mha_act_bits', type=int, default=32, help='MHA Layer Act bits(default:32)')

    parser.add_argument('--mha_act_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='MHA Layer Act rounding method (default: nearest)')

    parser.add_argument('--mha_act_per_channel', type=bool, default=False, help='MHA Layer Act lsq per_channel(default:False)')
    parser.add_argument('--mha_act_sig', type=int, default=23, help='MHA Layer Act Mantissa (default:23)')
    parser.add_argument('--mha_act_man', type=int, default=8, help='MHA Layer Act Significant (default:8)')

    #   Error:
    parser.add_argument('--mha_error_qmode', type=str, default='none',
                        choices=['adaptive', 'fixed', 'absmax', 'none'],
                        help='The quantization mode for mha Layer errors!')
    parser.add_argument('--mha_error_rep', type=str, default='fp', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='MHA Layer Error representation (default: int)')

    parser.add_argument('--mha_error_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='MHA Layer Error rounding method (default: nearest)')

    parser.add_argument('--mha_error_sig', type=int, default=8, help='MHA Layer error Significant (default:8)')
    parser.add_argument('--mha_error_man', type=int, default=23, help='MHA Layer error Mantissa (default:23)')
    parser.add_argument('--mha_error_scale', type=float, default=100000.0, help='MHA Layer Error scale (default: 100000)')



    # ---------MHA-LINEAR--------

    #     Act
    parser.add_argument('--mha_linear_act_qmode', type=str, default='none',
                        choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act', 'none'],
                        help='The quantization mode for MHA-Linear Layers activations!')
    parser.add_argument('--mha_linear_act_rep',
                        type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Act representation of MHA-Linear Layer(default: int)')

    parser.add_argument('--mha_linear_act_bits', type=int, default=32, help='MHA-Linear Layer Act bits(default:32)')

    parser.add_argument('--mha_linear_act_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='MHA-Linear Layer Act rounding method (default: nearest)')

    parser.add_argument('--mha_linear_act_per_channel', type=bool, default=False, help='MHA-Linear Layer Act lsq per_channel(default:False)')
    parser.add_argument('--mha_linear_act_sig', type=int, default=23, help='MHA-Linear Layer Act Mantissa (default:23)')
    parser.add_argument('--mha_linear_act_man', type=int, default=8, help='MHA-Linear Layer Act Significant (default:8)')



    #   Weights:
    parser.add_argument('--mha_linear_weight_qmode', type=str, default='none',
                        choices=['lsq_weight', 'sawb', 'dorefa_weight', 'minmax_weight', 'none'],
                        help='The quantization mode for MHA-Linear Layer weights!')
    parser.add_argument('--mha_linear_weight_rep',
                        type=str,
                        default='int',
                        choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Weight representation for MHA-Linear Layer(default: int)')
    parser.add_argument('--mha_linear_weight_bits', type=int, default=32, help='Weight bits for MHA-Linear Layer (default:32)')
    parser.add_argument('--mha_linear_weight_rounding',
                        type=str,
                        default='nearest',
                        choices=['nearest', 'stochastic'],
                        help='Weight rounding method for MHA-Linear Layer (default: nearest)')

    parser.add_argument('--mha_linear_weight_per_channel', type=bool, default=False, help='Weight lsq per_channel for MHA-Linear Layer(default:False)')
    parser.add_argument('--mha_linear_weight_sig', type=int, default=23, help='weight Mantissa for MHA-Linear Layer(default:23)')
    parser.add_argument('--mha_linear_weight_man', type=int, default=8, help='weight Significant for MHA-Linear Layer(default:8)')



    #   Error:
    parser.add_argument('--mha_linear_error_qmode', type=str, default='none',
                        choices=['adaptive', 'fixed', 'absmax', 'none'],
                        help='The quantization mode for MHA-Linear Layer errors!')
    parser.add_argument('--mha_linear_error_rep', type=str, default='fp', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='MHA-Linear Layer Error representation (default: int)')

    parser.add_argument('--mha_linear_error_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='MHA-Linear Layer Error rounding method (default: nearest)')

    parser.add_argument('--mha_linear_error_sig', type=int, default=8, help='MHA-Linear Layer error Significant (default:8)')
    parser.add_argument('--mha_linear_error_man', type=int, default=23, help='MHA-Linear Layer error Mantissa (default:23)')
    parser.add_argument('--mha_linear_error_scale', type=float, default=100000.0, help='MHA-Linear Layer Error scale (default: 100000)')


    # ---------Last Layer---------

    #     Act
    parser.add_argument('--last_act_qmode', type=str, default='none',
                        choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act', 'none'],
                        help='The quantization mode for Last Layer activations!')
    parser.add_argument('--last_act_rep',
                        type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Act representation of Last Layer(default: int)')

    parser.add_argument('--last_act_bits', type=int, default=32, help='Last Layer Act bits(default:32)')

    parser.add_argument('--last_act_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='Last Layer Act rounding method (default: nearest)')

    parser.add_argument('--last_act_per_channel', type=bool, default=False, help='Last Layer Act lsq per_channel(default:False)')
    parser.add_argument('--last_act_sig', type=int, default=23, help='Last Layer Act Mantissa (default:23)')
    parser.add_argument('--last_act_man', type=int, default=8, help='Last Layer Act Significant (default:8)')



    #   Weights:
    parser.add_argument('--last_weight_qmode', type=str, default='none',
                        choices=['lsq_weight', 'sawb', 'dorefa_weight', 'minmax_weight', 'none'],
                        help='The quantization mode for last Layer weights!')
    parser.add_argument('--last_weight_rep',
                        type=str,
                        default='int',
                        choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Weight representation for last Layer(default: int)')
    parser.add_argument('--last_weight_bits', type=int, default=32, help='Weight bits for last Layer (default:32)')
    parser.add_argument('--last_weight_rounding',
                        type=str,
                        default='nearest',
                        choices=['nearest', 'stochastic'],
                        help='Weight rounding method for last Layer (default: nearest)')

    parser.add_argument('--last_weight_per_channel', type=bool, default=False, help='Weight lsq per_channel for last Layer(default:False)')
    parser.add_argument('--last_weight_sig', type=int, default=23, help='weight Mantissa for last Layer(default:23)')
    parser.add_argument('--last_weight_man', type=int, default=8, help='weight Significant for last Layer(default:8)')



    #   Error:
    parser.add_argument('--last_error_qmode', type=str, default='none',
                        choices=['adaptive', 'fixed', 'absmax', 'none'],
                        help='The quantization mode for last Layer errors!')
    parser.add_argument('--last_error_rep', type=str, default='fp', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='last Layer Error representation (default: int)')

    parser.add_argument('--last_error_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='last Layer Error rounding method (default: nearest)')

    parser.add_argument('--last_error_sig', type=int, default=8, help='last Layer error Significant (default:8)')
    parser.add_argument('--last_error_man', type=int, default=23, help='last Layer error Mantissa (default:23)')
    parser.add_argument('--last_error_scale', type=float, default=100000.0, help='last Layer Error scale (default: 100000)')


    # ---------Layer Norm---------
    parser.add_argument('--ln_act_qmode', type=str, default='none',
                        choices=['lsq_act', 'pact', 'dorefa_act', 'minmax_act', 'none'],
                        help='The quantization mode for Layer Norm activations!')
    parser.add_argument('--ln_act_rep',
                        type=str, default='int', choices=['int', 'fp', 'rdx2', 'rdx4'],
                        help='Act representation of Layer Norm(default: int)')

    parser.add_argument('--ln_act_bits', type=int, default=32, help='Layer Norm Act bits(default:32)')

    parser.add_argument('--ln_act_rounding',
                        type=str, default='nearest', choices=['nearest', 'stochastic'],
                        help='Layer Norm Act rounding method (default: nearest)')

    parser.add_argument('--ln_act_per_channel', type=bool, default=False, help='Layer Norm Act lsq per_channel(default:False)')
    parser.add_argument('--ln_act_sig', type=int, default=23, help='Layer Norm Act Mantissa (default:23)')
    parser.add_argument('--ln_act_man', type=int, default=8, help='Layer Norm Act Significant (default:8)')



    # Others:
    parser.add_argument('--optim_qmode', type=str, default='None', choices=['None', 'bnb'],
                        help='Optimizer Quantization')
    parser.add_argument('--optim_bits', type=int, default=32, choices=[32, 8],
                        help='Optimizer bits(default:32)')

    parser.add_argument('--first_layer_quant', type=str2bool, default=False,
                        help='Weather we want to quantize the first layer or not')

    parser.add_argument('--last_layer_quant', type=str2bool, default=False,
                        help='Weather we want to quantize the last layer or not')

    parser.add_argument('--shortcut_quant', type=str2bool, default=False,
                        help='Weather we want to quantize the shortcut or not')

    return parser


def quant_args_parser(parser):

    parser = quant_parser(parser)

    args = parser.parse_args()

    # TODO: Add error_bits to the args
    if args.error_rep != 'int':
        args.error_bits = args.error_sig + args.error_man + 1
    if args.bn_error_rep != 'int':
        args.bn_error_bits = args.bn_error_sig + args.bn_error_man + 1
    if args.first_error_rep != 'int':
        args.first_error_bits = args.first_error_sig + args.first_error_man + 1
    if args.last_error_rep != 'int':
        args.last_error_bits = args.last_error_sig + args.last_error_man + 1
    if args.mha_error_rep != 'int':
        args.mha_error_bits = args.mha_error_sig + args.mha_error_man + 1
    if args.fc_error_rep != 'int':
        args.fc_error_bits = args.fc_error_sig + args.fc_error_man + 1


    return args



def args_parser():

    parser = argparse.ArgumentParser(description='PyTorch Quantization Training for CIFAR-10')


    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenet'],
                        help='Model (default: resnet18)')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                         help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                         help='number of epochs to train (default: 100)')
    parser.add_argument('--lr-epochs', type=int, default=30,
                         help='learning rate scheduler (default: 30)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')

    args = quant_args_parser(parser)

    return args


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr-epochs epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    print('Current Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
