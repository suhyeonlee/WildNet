"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import random


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR50V3PlusD',
                    help='Network architecture.')
parser.add_argument('--dataset', nargs='*', type=str, default=['gtav'],
                    help='a list of datasets; cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['bdd100k'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--wild_dataset', nargs='*', type=str, default=['imagenet'],
                    help='a list consists of imagenet')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet-50',
                    help='trunk model, can be: resnet-50 (default)')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')

parser.add_argument('--fs_layer', nargs='*', type=int, default=[0,0,0,0,0],
                    help='0: None, 1: AdaIN')
parser.add_argument('--lambda_cel', type=float, default=0.0,
                    help='lambda for content extension learning loss')
parser.add_argument('--lambda_sel', type=float, default=0.0,
                    help='lambda for style extension learning loss')
parser.add_argument('--lambda_scr', type=float, default=0.0,
                    help='lambda for semantic consistency regularization loss')
parser.add_argument('--cont_proj_head', type=int, default=0, 
                    help='number of output channels of content projection head')
parser.add_argument('--wild_cont_dict_size', type=int, default=0,
                    help='wild-content dictionary size')

parser.add_argument('--use_fs', action='store_true', default=False,
                    help='Automatic setting from fs_layer. feature stylization with wild dataset')
parser.add_argument('--use_scr', action='store_true', default=False,
                    help='Automatic setting from lambda_scr')
parser.add_argument('--use_sel', action='store_true', default=False,
                    help='Automatic setting from lambda_sel')
parser.add_argument('--use_cel', action='store_true', default=False,
                    help='Automatic setting from lambda_cel')

args = parser.parse_args()

random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=args.world_size,
                                    rank=args.local_rank)
# torch.distributed.init_process_group(backend='nccl',
#                                     init_method=args.dist_url,
#                                     world_size=args.world_size,
#                                     rank=args.local_rank)

for i in range(len(args.fs_layer)):
    if args.fs_layer[i] == 1:
        args.use_fs = True

if args.lambda_cel > 0:
    args.use_cel = True
if args.lambda_sel > 0:
    args.use_sel = True
if args.lambda_scr > 0:
    args.use_scr = True

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    train_source_loader, val_loaders, train_wild_loader, train_obj, extra_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux, args.cont_proj_head, args.wild_cont_dict_size)

    optim, scheduler = optimizer.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_source_loader)
            i = iter_per_epoch * epoch
            epoch = epoch + 1
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        i = train(train_source_loader, train_wild_loader, net, optim, epoch, writer, scheduler, args.max_iter)
        train_source_loader.sampler.set_epoch(epoch + 1)
        train_wild_loader.sampler.set_epoch(epoch + 1)

        if args.local_rank == 0:
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler, None, None, [],
                        writer, epoch, "None", None, i, save_pth=True)

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                train_source_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()
        
        epoch += 1
    
    # Validation after epochs
    if len(val_loaders) == 1:
        # Run validation only one time - To save models
        for dataset, val_loader in val_loaders.items():
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
    else:
        if args.local_rank == 0:
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler, None, None, [],
                        writer, epoch, "None", None, i, save_pth=True)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)


def train(source_loader, wild_loader, net, optim, curr_epoch, writer, scheduler, max_iter):
    """
    Runs the training loop per epoch
    source_loader: Source data loader for train
    wild_loader: Wild data loader for train
    net: thet network
    optim: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(source_loader)

    wild_loader_iter = enumerate(wild_loader)

    for i, data in enumerate(source_loader):
        if curr_iter >= max_iter:
            break

        inputs, gts, _, aux_gts = data

        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

        batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
            input, gt, aux_gt = ingredients

            _, inputs_wild = next(wild_loader_iter)
            input_wild = inputs_wild[0]

            start_ts = time.time()

            img_gt = None
            input, gt = input.cuda(), gt.cuda()
            input_wild = input_wild.cuda()

            optim.zero_grad()
            outputs = net(x=input, gts=gt, aux_gts=aux_gt, x_w=input_wild, apply_fs=args.use_fs)
            
            outputs_index = 0
            main_loss = outputs[outputs_index]
            outputs_index += 1
            aux_loss = outputs[outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            if args.use_fs:
                if args.use_cel:
                    cel_loss = outputs[outputs_index]
                    outputs_index += 1
                    total_loss = total_loss + (args.lambda_cel * cel_loss)
                else:
                    cel_loss = 0
                
                if args.use_sel:
                    sel_loss_main = outputs[outputs_index]
                    outputs_index += 1
                    sel_loss_aux = outputs[outputs_index]
                    outputs_index += 1
                    total_loss = total_loss + args.lambda_sel * (sel_loss_main + (0.4 * sel_loss_aux))
                else:
                    sel_loss_main = 0
                    sel_loss_aux = 0

                if args.use_scr:
                    scr_loss_main = outputs[outputs_index]
                    outputs_index += 1
                    scr_loss_aux = outputs[outputs_index]
                    outputs_index += 1
                    total_loss = total_loss + args.lambda_scr * (scr_loss_main + (0.4 * scr_loss_aux))
                else:
                    scr_loss_main = 0
                    scr_loss_aux = 0


            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)
            
            total_loss.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss

            if args.local_rank == 0:
                if i % 50 == 49:
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(source_loader), curr_iter, train_total_loss.avg,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)

                    logging.info(msg)
                    
                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg), curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):

        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    return val_loss.avg


if __name__ == '__main__':
    main()
