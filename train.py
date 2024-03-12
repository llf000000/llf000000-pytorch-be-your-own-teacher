import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os
import shutil
import argparse
import time
import logging
import math

import models.resnet as models
from dataset.data import *

import torchvision.models.utils as utils
from tensorboardX import SummaryWriter 
import numpy as np

def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data-dir', default='/home/xxx/cifar100', type=str,
                        help='the diretory to save cifar100 dataset')
    parser.add_argument('arch', metavar='ARCH', default='multi_resnet50_kd',
                        help='model architecture')
    '''
    'arch'：这是参数的名称，表示模型的架构。
    metavar='ARCH'：这是参数的元变量名称，用于在帮助信息中显示参数的名称。
    default='multi_resnet50_kd'：这是参数的默认值，如果用户没有在命令行中指定该参数，将使用默认值。
    help='model architecture'：这是参数的帮助信息，它描述了该参数的作用和用法。
    在脚本运行时，用户可以通过命令行指定模型的架构。例如，可以使用以下命令指定模型架构为 "multi_resnet50_kd"：
        python script.py multi_resnet50_kd
    如果用户没有在命令行中指定模型架构，则将使用默认值 "multi_resnet50_kd"。该参数的值可以在代码中通过 args.arch 访问
    '''
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    '''通过添加 --resume 参数，用户可以在命令行中指定一个检查点文件的路径，用于恢复之前训练的模型状态。
    如果用户不指定 --resume 参数，则默认值为空字符串。
    python train.py train multi_resnet18_kd --resume ./save_checkpoints/multi_resnet18_kd/checkpoint_latest.pth.tar
    '''
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--summary-folder', default='runs_alpha01/', type=str,
                        help='folder to save the summary')
    parser.add_argument('--eval-every', default=1000, type=int,
                        help='evaluate model every (default: 1000) iterations')

    #kd parameter
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)  # save_folder save_checkpoints/      'arch', metavar='ARCH', default='multi_resnet50_kd'
    '''
    --save-folder', default='save_checkpoints/'
        'arch', metavar='ARCH', default='multi_resnet50_kd'
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    '''
    假设 save_path 的值为 /home/user/logs，args.cmd 的值为 'train'，那么拼接后的日志文件路径将是
     /home/user/logs/log_train.txt。
    '''
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)
    
    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)
    
    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        run_test(args)


def run_test(args):
    writer = SummaryWriter(args.summary_folder)
    if args.dataset == 'cifar100': 
        model = models.__dict__[args.arch](num_classes=100)
    else:
        raise NotImplementedError
    model = torch.nn.DataParallel(model).to(args.device)

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))
            exit()
    
    cudnn.benchmark = True # 启用 CuDNN 的自动调优功能，以提高训练和测试的性能。

    #load dataset
    if args.dataset == 'cifar100':
        test_loader = prepare_cifar100_test_dataset(data_dir=args.data_dir, batch_size=args.batch_size, 
                                                        num_workers=args.workers)
    else:
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss().to(args.device)
    validate(args, test_loader, model, criterion)

def run_training(args):
    writer = SummaryWriter(args.summary_folder) # '--summary-folder', default='runs_alpha01/',
    if args.dataset == 'cifar100': # '--dataset', '-d', type=str, default='cifar100'
        model = models.__dict__[args.arch](num_classes=100) # 'arch', metavar='ARCH', default='multi_resnet50_kd'
    else:
        raise NotImplementedError # 如果指定的数据集不是cifar100，则抛出NotImplementedError异常。
    model = torch.nn.DataParallel(model).to(args.device) # 将模型放在指定的设备上进行并行处理。使用 torch.nn.DataParallel 将模型包装成一个可以利用多个 GPU 进行并行计算的模型。
    best_prec1 = 0  # 用于跟踪训练过程中的最高精度。

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))
    
    cudnn.benchmark = True # ：启用cuDNN的自动调优器，以找到最适合当前配置的卷积算法，从而提高训练速度。
    if args.dataset == 'cifar100':
        train_loader = prepare_cifar100_train_dataset(data_dir=args.data_dir, batch_size=args.batch_size, 
                                                        num_workers=args.workers) # from dataset.data import *
        test_loader = prepare_cifar100_test_dataset(data_dir=args.data_dir, batch_size=args.batch_size, 
                                                        num_workers=args.workers)
    else:
        raise NotImplementedError
   
    criterion = nn.CrossEntropyLoss().to(args.device) # 当你使用这个损失函数计算一个批次的数据时，它默认计算的是批次中每个样本的损失，并返回这些损失的平均值。
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)


    end = time.time()
    model.train()
    step = 0
    for current_epoch in range(args.start_epoch, args.epoch):
        # 对于每个纪元，初始化一系列的计量器（用于跟踪损失、精度等），调整学习率，然后遍历训练数据。
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        middle1_losses = AverageMeter()
        middle2_losses = AverageMeter()
        middle3_losses = AverageMeter()
        losses1_kd = AverageMeter()
        losses2_kd = AverageMeter()
        losses3_kd = AverageMeter()
        feature_losses_1 = AverageMeter()
        feature_losses_2 = AverageMeter()
        feature_losses_3 = AverageMeter()
        total_losses = AverageMeter()
        middle1_top1 = AverageMeter()
        middle2_top1 = AverageMeter()
        middle3_top1 = AverageMeter()

        adjust_learning_rate(args, optimizer, current_epoch)
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end) # 更新data_time以记录从上一批数据处理结束到当前批数据开始处理的时间。这有助于分析数据加载的效率。
            
            target = target.squeeze().long().to(args.device) # 将目标标签（target）通过squeeze()方法去除不必要的维度并转换为长整型，然后移动到指定的设备（如GPU）。输入数据（input）也被移动到指定的设备。
            input = input.to(args.device)

            output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)
            
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            '''
            input.size(0): 这部分获取当前批次的样本数量。在PyTorch中，input是一个多维张量，
            其中size(0)返回第一个维度的大小，即批次中的样本数。这个信息对于计算平均损失非常重要，
            因为你可能想要知道整个训练集上的平均损失而不仅仅是单个批次的。
            '''
            middle1_loss = criterion(middle_output1, target)
            middle1_losses.update(middle1_loss.item(), input.size(0))
            middle2_loss = criterion(middle_output2, target)
            middle2_losses.update(middle2_loss.item(), input.size(0))
            middle3_loss = criterion(middle_output3, target)
            middle3_losses.update(middle3_loss.item(), input.size(0))

            temp4 = output / args.temperature
            temp4 = torch.softmax(temp4, dim=1)
            
            
            loss1by4 = kd_loss_function(middle_output1, temp4.detach(), args) * (args.temperature**2)
            losses1_kd.update(loss1by4, input.size(0))
            
            loss2by4 = kd_loss_function(middle_output2, temp4.detach(), args) * (args.temperature**2)
            losses2_kd.update(loss2by4, input.size(0))
            
            loss3by4 = kd_loss_function(middle_output3, temp4.detach(), args) * (args.temperature**2)
            losses3_kd.update(loss3by4, input.size(0))
            
            feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
            feature_losses_1.update(feature_loss_1, input.size(0))
            feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
            feature_losses_2.update(feature_loss_2, input.size(0))
            feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 
            feature_losses_3.update(feature_loss_3, input.size(0))

            total_loss = (1 - args.alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        args.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                        args.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
            total_losses.update(total_loss.item(), input.size(0))
            
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))

            middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
            middle2_top1.update(middle2_prec1[0], input.size(0))
            middle3_prec1 = accuracy(middle_output3.data, target, topk=(1,))
            middle3_top1.update(middle3_prec1[0], input.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                writer.add_scalar('loss', losses.avg, step)
                writer.add_scalar('middle1_loss', middle1_losses.avg, step)
                writer.add_scalar('middle2_loss', middle2_losses.avg, step)
                writer.add_scalar('middle3_loss', middle3_losses.avg, step)
                writer.add_scalar('loss1_kd', losses1_kd.avg, step)
                writer.add_scalar('loss2_kd', losses2_kd.avg, step)
                writer.add_scalar('loss3_kd', losses3_kd.avg, step)
                writer.add_scalar('total_loss', total_losses.avg, step)
                writer.add_scalar('accuracy', top1.avg, step)
                writer.add_scalar('middle1_acc', middle1_top1.avg, step)
                writer.add_scalar('middle2_acc', middle2_top1.avg, step)
                writer.add_scalar('middle3_acc', middle3_top1.avg, step)
                writer.add_scalar('feature_loss_1', feature_losses_1.avg, step)
                writer.add_scalar('feature_loss_2', feature_losses_2.avg, step)
                writer.add_scalar('feature_loss_3', feature_losses_3.avg, step)
                
                step += 1
                logging.info("Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                current_epoch,
                                i,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=total_losses,
                                top1=top1)
                ) 
        prec1 = validate(args, test_loader, model, criterion, writer, current_epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print("best: ", best_prec1)
        checkpoint_path = os.path.join(args.save_path, 'checkpoint_{:05d}.pth.tar'.format(current_epoch))
        save_checkpoint({
            'epoch': current_epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path) # 
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path, 'checkpoint_latest.pth.tar'))
        torch.cuda.empty_cache() # torch.cuda.empty_cache()释放未使用的缓存，这可以帮助减少PyTorch在GPU上的内存占用。在长时间运行或多次迭代训练过程中，这有助于防止内存溢出。


        
def validate(args, test_loader, model, criterion, writer=None, current_epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    middle1_losses = AverageMeter()
    middle2_losses = AverageMeter()
    middle3_losses = AverageMeter()
    top1 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        target = target.squeeze().long().to(args.device)
        input = input.to(args.device)

        output, middle_output1, middle_output2, middle_output3, \
        final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)
            
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        middle1_loss = criterion(middle_output1, target)
        middle1_losses.update(middle1_loss.item(), input.size(0))
        middle2_loss = criterion(middle_output2, target)
        middle2_losses.update(middle2_loss.item(), input.size(0))
        middle3_loss = criterion(middle_output3, target)
        middle3_losses.update(middle3_loss.item(), input.size(0))
            
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
        middle1_top1.update(middle1_prec1[0], input.size(0))
        middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
        middle2_top1.update(middle2_prec1[0], input.size(0))
        middle3_prec1 = accuracy(middle_output3.data, target, topk=(1,))
        middle3_top1.update(middle3_prec1[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    logging.info("Loss {loss.avg:.3f}\t"
                 "Prec@1 {top1.avg:.3f}\t"
                 "Middle1@1 {middle1_top1.avg:.3f}\t"
                 "Middle2@1 {middle2_top1.avg:.3f}\t"
                 "Middle3@1 {middle3_top1.avg:.3f}\t".format(
                    loss=losses,
                    top1=top1,
                    middle1_top1=middle1_top1,
                    middle2_top1=middle2_top1,
                    middle3_top1=middle3_top1))
    
    if writer is not None:
        writer.add_scalar('val_loss', losses.avg, current_epoch)
        writer.add_scalar('val_middle1_loss', middle1_losses.avg, current_epoch)
        writer.add_scalar('val_middle2_loss', middle2_losses.avg, current_epoch)
        writer.add_scalar('val_middle3_loss', middle3_losses.avg, current_epoch)
        writer.add_scalar('val_accuracy', top1.avg, current_epoch)
        writer.add_scalar('val_middle1_acc', middle1_top1.avg, current_epoch)
        writer.add_scalar('val_middle2_acc', middle2_top1.avg, current_epoch)
        writer.add_scalar('val_middle3_acc', middle3_top1.avg, current_epoch)
        logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    
        
    model.train()
    return top1.avg

def kd_loss_function(output, target_output,args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # 当你使用这个损失函数计算一个批次的数据时，它默认计算的是批次中每个样本的损失，并返回这些损失的平均值。
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    if args.warm_up and (epoch < 1):
        lr = 0.01
    elif 75 <= epoch < 130:
        lr = args.lr * (args.step_ratio ** 1)
    elif 130 <= epoch < 180:
        lr = args.lr * (args.step_ratio ** 2)
    elif epoch >=180:
        lr = args.lr * (args.step_ratio ** 3)
    else:
        lr = args.lr

    
    logging.info('Epoch [{}] learning rate = {}'.format(epoch, lr))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    '''
    通过比较模型的Top-k预测与真实标签来计算准确率，能够同时处理多种Top-k准确率的计算，并返回一个包含这些准确率的列表
    '''
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # 调用topk方法获取每个样本预测概率最高的maxk个类别的索引。pred的形状为[批次大小, maxk]。
    pred = pred.t() # 将pred矩阵转置，使其形状变为[maxk, 批次大小]。这样做是为了便于后续与目标标签进行比较
    correct = pred.eq(target.view(1, -1).expand_as(pred))   # 使用eq方法比较预测和目标是否相等，得到一个布尔矩阵correct，表示每个预测是否正确

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) # 得到正确预测的数量
        res.append(correct_k.mul(100.0 / batch_size)) # 计算准确率的百分比
    
    return res

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename) 
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.path.tar'))  # 使用shutil.copyfile方法将当前保存的检查点复制一份到相同目录下，并命名为model_best.path.tar

if __name__ == '__main__':
    main()
