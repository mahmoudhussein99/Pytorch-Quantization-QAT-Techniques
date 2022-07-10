import random
import os
import argparse
import sys

import numpy as np
import wandb
import torch
import torchvision
import torchvision.models.quantization as models

# from ../ import *
# import sys
sys.path.append('../../')
sys.path.append('../../cpp')
sys.path.append('../../experiments')
sys.path.append('../../opt_quant')


from experiments.utils import get_q_modules, get_sgd_optimizer, get_bn_modules, get_first_layer_modules
from opt_quant import *
from experiments.oxfordPets.q_model import ResNet18, MobileNetV2,ResNet50
from experiments.utils import adjust_learning_rate, progress_bar, quant_args_parser
from experiments.oxfordPets.utils import data_generator

best_acc = 0

# Training
def train(epoch, trainloader, net, optimizer, criterion, args, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    epoch_accs = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        curr_iter = epoch * len(trainloader) + batch_idx

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            print('NAN loss!')
            wandb.log({'test_acc': -1})
            exit(2)

        # L2 regularization for PACT
        for name, param in net.named_parameters():
            if "alpha" in name:
                loss += (args.pact_reg*torch.pow(param, 2)[0])

        if torch.isnan(loss):
            print('NAN from regularized loss!')
            wandb.log({'test_acc_regularized': -1})
            exit(2)

        loss.backward()

        # Adaptive gradScale
        for name, param in net.named_parameters():
            if 'adaptive_scale' in name:
                if param.grad.data > 0:
                    param.data *= 2.0
                elif param.grad.data < 0:
                    param.data *= 0.5
                param.grad = None

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        batch_total = targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
        epoch_accs.append(100. * batch_correct / batch_total)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1),
                        epoch_accs[-1],
                        batch_correct, batch_total))

        wandb.log({
            'train_batch_acc': epoch_accs[-1],
            'epoch': epoch,
            'batch': batch_idx,
            'train_step': epoch * len(trainloader) + batch_idx,
            'train_batch_loss': loss.item(),
        })
    wandb.log({
        'train_epoch_acc': np.mean(epoch_accs),
        'epoch': epoch,
    })


def test(epoch, testloader, net, criterion, device):
    global best_acc
    net.eval()
    test_loss = 0
    epoch_accs = []
    output_features=37
    confusion_matrix = torch.zeros(output_features, output_features)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, preds =torch.max(outputs,1)
            for t,p in zip(targets.view(-1),preds.view(-1)):
                confusion_matrix[t.long(),p.long()]+=1
            batch_total = targets.size(0)
            batch_correct = predicted.eq(targets).sum().item()
            epoch_accs.append(100. * batch_correct / batch_total)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), epoch_accs[-1], batch_correct, batch_total))

            wandb.log({
                'test_batch_acc': epoch_accs[-1],
                'epoch': epoch,
                'batch': batch_idx,
                'test_batch_loss': loss.item(),
            })

    epoch_accs = np.array(epoch_accs)

    # Save checkpoint.
    #acc = np.mean(epoch_accs)
    acc=100*torch.mean(confusion_matrix.diag()/confusion_matrix.sum(1))
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        dir_ = './checkpoint/{}'.format(wandb.run.name)
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
        torch.save(state, '{}/ckpt.pth'.format(dir_))
        best_acc = acc

    wandb.log({
        'test_epoch_acc': acc,
        'epoch': epoch,
        'test_best_acc': best_acc,
    })


def train_model(args, device):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, testloader = data_generator(args)

    # Model
    print('==> Building model...')
    w_qmodule, act_qmodule, err_qmodule = get_q_modules(args)
    bn_w_qmodule, bn_act_qmodule, bn_err_qmodule = get_bn_modules(args)
    first_w_qmodule, first_act_qmodule, first_err_qmodule = get_first_layer_modules(args)
    if args.model == 'resnet18':
        net = ResNet18(w_qmodule, act_qmodule, err_qmodule,
                       bn_w_qmodule, bn_act_qmodule, bn_err_qmodule,
                       first_w_qmodule, first_act_qmodule, first_err_qmodule,
                   last_layer_quant=args.last_layer_quant,
                   bn=args.bn,
                   shortcut_quant=args.shortcut_quant)
        #M_EDIT
        pretrained=args.pretrained
        if(pretrained):
            print('lets see the model params and try to copy them')
            model_pretrained=models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)

            for name, param in model_pretrained.named_parameters():
                    #print('name: ', name)
                    #print(type(param))
                    #print('param.shape: ', param.shape)
                    #print('param.requires_grad: ', param.requires_grad)
                    #     print(param)
                    #print('=====')
                    for name_net, param_net in net.named_parameters():
                            #print('name: ', name_net)
                            #print(type(param_net))
                            #print('param.shape: ', param_net.shape)
                            #print('param.requires_grad: ', param_net.requires_grad)
                            #     print(param_net)
                            #print('=====')
                            if(name_net==name and param_net.shape==param.shape):
                                    print('match found')
                                    param_net.data=param.data
                                    print(param_net)
                                    print('should be success!!')
                                    print('======')
            print('done loading pretrained params')

    #M_EDIT
    elif args.model == 'resnet50':
        pretrained=args.pretrained
        net=ResNet50(w_qmodule, act_qmodule, err_qmodule,
                       bn_w_qmodule, bn_act_qmodule, bn_err_qmodule,
                       first_w_qmodule, first_act_qmodule, first_err_qmodule,
                   last_layer_quant=args.last_layer_quant,
                   bn=args.bn,
                   shortcut_quant=args.shortcut_quant)
        
        if(pretrained):
            print("next step TODO")
        sys.exit()
            

    elif args.model == 'mobilenet':
        net = MobileNetV2(w_qmodule, act_qmodule, err_qmodule,
                          bn_w_qmodule, bn_act_qmodule, bn_err_qmodule,
                          first_w_qmodule, first_act_qmodule, first_err_qmodule,
                       last_layer_quant=args.last_layer_quant,
                       bn=args.bn,
                       shortcut_quant=args.shortcut_quant)

    net = net.to(device)
    #print('lets see the model params and try to copy them')
    #model_pretrained=models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)

    #for name, param in model_pretrained.named_parameters():
            #print('name: ', name)
            #print(type(param))
            #print('param.shape: ', param.shape)
            #print('param.requires_grad: ', param.requires_grad)
            #     print(param)
            #print('=====')
            #for name_net, param_net in net.named_parameters():
                    #print('name: ', name_net)
                    #print(type(param_net))
                    #print('param.shape: ', param_net.shape)
                    #print('param.requires_grad: ', param_net.requires_grad)
                    #     print(param_net)
                    #print('=====')
                    #if(name_net==name and param_net.shape==param.shape):
                        #print('match found')
                        #param_net.data=param.data
                        #print(param_net)
                        #print('should be success!!')


    #print('done loading pretrained params')
    #exit()
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = get_sgd_optimizer(net, args)

    #M_EDIT
    # wandb.init(project='state_quantization_neurips22', entity='saleh_projects')
    wandb.init(project="OxfordPets-QAT8", entity="mhussein") 
    #wandb.init(project='state_quantization_TL_QAT', entity='quantization-tl-team')
    wandb.config.update(args)

    print(net)
    print(optimizer)
    for epoch in range(start_epoch, args.epochs):
        train(epoch, trainloader, net, optimizer, criterion, args, device)
        test(epoch, testloader, net, criterion, device)
        adjust_learning_rate(args, optimizer, epoch)


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Quantization Training for Oxford Pets')


    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    # parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenet','resnet50'],
    #                     help='Model (default: resnet18)')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenet','resnet50'],
                        help='Model (default: resnet18)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                         help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                         help='number of epochs to train (default: 150)')
    parser.add_argument('--lr-epochs', type=int, default=50,
                         help='learning rate scheduler (default: 50)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 0.0001)')
    #M_EDIT
    parser.add_argument('--pretrained', type=bool, default=True, metavar='P',
                        help='pretrained values (default: True)')
    
    return parser


if __name__ == '__main__':
    print("let's start some fun")
    parser = args_parser()
    args = quant_args_parser(parser)
    if 'stochastic' not in args.error_rounding:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and torch.cuda.device_count() > 1:
        torch.cuda.set_device(random.randint(0, torch.cuda.device_count() - 1))

    train_model(args, device)
