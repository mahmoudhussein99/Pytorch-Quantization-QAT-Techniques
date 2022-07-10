# This script is used to continue trianing on Ault. This will create an "last_ault_checkpoint" directory and
# copy the last checkpoint on that directory. Then each time, it will resume from that checkpoint and continue trianing!

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import wandb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

from experiments.utils import get_q_modules, str2bool, get_bn_modules, get_first_layer_modules, tuple_float_type

from experiments.imageNet import q_model as models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--resume_from_ault', action='store_true',
                    help='If this is true, we load from last_ault_checkpoint directory!')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:9919', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


# ---------General---------

# Weight
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


# Activation
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

parser.add_argument('--act_mode', type=str, default='unsigned',
                    choices=['unsigned', 'signed'],
                    help='Act dorefa mode(default:unsigned)')

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
parser.add_argument('--bn', type=str, default='BN', choices=['BN', 'RangeBN', 'OriginalRangeBN'],
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



# Others:
parser.add_argument('--last_layer_quant', type=str2bool, default=False,
                    help='Weather we want to quantize the last layer or not')

parser.add_argument('--first_layer_quant', type=str2bool, default=False,
                    help='Weather we want to quantize the first layer or not')

parser.add_argument('--shortcut_quant', type=str2bool, default=True,
                    help='Weather we want to quantize the shortcut or not')




best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.resume_from_ault and os.path.exists('last_ault_checkpoint/checkpoint.pth.tar'):
        args.resume = 'last_ault_checkpoint'
    elif not os.path.exists('last_ault_checkpoint/'):
        os.mkdir('last_ault_checkpoint/')

    if args.error_rep != 'int':
        args.error_bits = args.error_sig + args.error_man + 1
    if args.first_error_rep != 'int':
        args.first_error_bits = args.first_error_sig + args.first_error_man + 1
    if args.bn_error_rep != 'int':
        args.bn_error_bits = args.bn_error_sig + args.bn_error_man + 1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    print('seed fixed!')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: ', ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    print('Main worker!')
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        wandb.init(project='state_quantization_neurips22', entity='saleh_projects')
        wandb.config.update(args)


    # create model
    print("=> creating model '{}'".format(args.arch))
    w_qmodule, act_qmodule, err_qmodule = get_q_modules(args)
    bn_w_qmodule, bn_act_qmodule, bn_err_qmodule = get_bn_modules(args)
    first_w_qmodule, first_act_qmodule, first_err_qmodule = get_first_layer_modules(args)
    model = models.__dict__[args.arch](
        first_w_qmodule=first_w_qmodule, first_act_qmodule=first_act_qmodule, first_err_qmodule=first_err_qmodule,
        w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule,
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
        bn=args.bn,
    )

    print(model)


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(args.resume, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, epoch=-1)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args, epoch) for now, we do not neeed to validate the model!
        acc1 = best_acc1 + 1 #to make sure that we save each checkpoint as the best one

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
            wandb.log(
                {
                    'epoch': epoch,
                    "top1_acc": top1.avg.item(),
                    "top5_acc": top5.avg.item(),
                }
            )

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    dir_ = 'checkpoints/{}'.format(wandb.run.name)
    ault_dir_ = 'last_ault_checkpoint'
    if not os.path.isdir(dir_):
        os.mkdir(dir_)

    torch.save(state, os.path.join(dir_, filename))
    shutil.copyfile(os.path.join(dir_, filename),  os.path.join(ault_dir_, 'checkpoint.pth.tar')) #copy to the last_ault_checkpoint directory
    if is_best:
        shutil.copyfile( os.path.join(dir_, filename),  os.path.join(dir_, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints/')
    main()



