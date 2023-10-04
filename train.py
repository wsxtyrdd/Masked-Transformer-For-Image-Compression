import os
import sys
from os.path import join

sys.path.append("/media/D/wangsixian/MT")

import configargparse
import time
from datetime import datetime
import math
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import shutil

torch.backends.cudnn.benchmark = False
from data.datasets import get_loader, get_dataset
from loss.distortion import Distortion
from utils import logger_configuration, load_weights, AverageMeter, save_checkpoint, worker_init_fn_seed
from net.mit import MaskedImageModelingTransformer


def train_one_epoch(epoch, net, train_loader, test_loader, optimizer,
                    device, logger):
    local_rank = torch.distributed.get_rank() if config.multi_gpu else 0
    best_loss = float("inf")
    mse_loss_wrapper = Distortion("MSE").to(device)
    ms_ssim_loss_wrapper = Distortion("MS-SSIM").to(device)
    elapsed, losses, psnrs, ms_ssim_dbs, bpps = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, losses, psnrs, ms_ssim_dbs, bpps]
    global global_step
    for batch_idx, input_image in enumerate(train_loader):
        net.train()
        B, C, H, W = input_image.shape
        num_pixels = B * H * W
        input_image = input_image.to(device)
        optimizer.zero_grad()
        global_step += 1
        start_time = time.time()
        results = net(input_image)
        bpp_loss = torch.log(results["likelihoods"]).sum() / (-math.log(2) * num_pixels)
        # if config.distortion_metric == "MSE":
        mse_loss = mse_loss_wrapper(results["x_hat"], input_image)
        tot_loss = config.lambda_value * 255 ** 2 * mse_loss + bpp_loss
        # elif config.distortion_metric == "MS-SSIM":
        #     ssim_loss = ms_ssim_loss_wrapper(results["x_hat"], input_image)
        # tot_loss = config.lambda_value * ssim_loss + bpp_loss
        tot_loss.backward()

        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.clip_max_norm)
        optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(tot_loss.item())
        bpps.update(bpp_loss.item())

        mse_val = mse_loss_wrapper(results["x_hat"], input_image).detach().item()
        psnr = 10 * (np.log(1. / mse_val) / np.log(10))
        psnrs.update(psnr.item())

        ms_ssim_val = ms_ssim_loss_wrapper(results["x_hat"], input_image).detach().item()
        ms_ssim_db = -10 * (np.log(ms_ssim_val) / np.log(10))
        ms_ssim_dbs.update(ms_ssim_db.item())

        if (global_step % config.print_every) == 0 and local_rank == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log_info = [
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'BPP {bpps.val:.2f} ({bpps.avg:.2f})',
                f'MS-SSIM {ms_ssim_dbs.val:.2f} ({ms_ssim_dbs.avg:.2f})',
                f'Epoch {epoch}'
            ]
            log = (' | '.join(log_info))
            logger.info(log)
            if config.wandb:
                log_dict = {"PSNR": psnrs.avg,
                            "MS-SSIM": ms_ssim_dbs.avg,
                            "BPP": bpps.avg,
                            "loss": losses.avg,
                            "Step": global_step,
                            }
                wandb.log(log_dict, step=global_step)
            for i in metrics:
                i.clear()

        if (global_step + 1) % config.test_every == 0 and local_rank == 0:
            loss = test(net, test_loader, device, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    is_best,
                    workdir
                )

        if (global_step + 1) % config.save_every == 0 and local_rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict()
                },
                False,
                workdir,
                filename='EP{}.pth.tar'.format(epoch)
            )


def test(net, test_loader, device, logger):
    with torch.no_grad():
        mse_loss_wrapper = Distortion("MSE").to(device)
        ms_ssim_loss_wrapper = Distortion("MS-SSIM").to(device)
        elapsed, losses, psnrs, ms_ssim_dbs, bpps = [AverageMeter() for _ in range(5)]
        global global_step
        for batch_idx, input_image in enumerate(test_loader):
            net.eval()
            B, C, H, W = input_image.shape
            num_pixels = B * H * W
            input_image = input_image.to(device)
            start_time = time.time()
            # crop and pad
            p = 64
            new_H = (H + p - 1) // p * p
            new_W = (W + p - 1) // p * p
            padding_left = (new_W - W) // 2
            padding_right = new_W - W - padding_left
            padding_top = (new_H - H) // 2
            padding_bottom = new_H - H - padding_top
            input_image_pad = F.pad(
                input_image,
                (padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            results = net.module.inference(input_image_pad)
            results["x_hat"] = F.pad(
                results["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
            )

            mse_loss = mse_loss_wrapper(results["x_hat"], input_image)
            bpp_loss = torch.log(results["likelihoods"]).sum() / (-math.log(2) * num_pixels)
            tot_loss = config.lambda_value * 255 ** 2 * mse_loss + bpp_loss
            elapsed.update(time.time() - start_time)
            losses.update(tot_loss.item())
            bpps.update(bpp_loss.item())

            mse_val = mse_loss.item()
            psnr = 10 * (np.log(1. / mse_val) / np.log(10))
            psnrs.update(psnr.item())

            ms_ssim_val = ms_ssim_loss_wrapper(results["x_hat"], input_image).mean().item()
            ms_ssim_db = -10 * (np.log(ms_ssim_val) / np.log(10))
            ms_ssim_dbs.update(ms_ssim_db.item())

            log_info = [
                f'Step [{(batch_idx + 1)}/{test_loader.__len__()}]',
                f'Loss ({losses.avg:.3f})',
                f'Time {elapsed.val:.2f}',
                f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'BPP {bpps.val:.2f} ({bpps.avg:.4f})',
                f'MS-SSIM {ms_ssim_dbs.val:.2f} ({ms_ssim_dbs.avg:.2f})',
            ]
            log = (' | '.join(log_info))
            logger.info(log)
            if config.wandb and local_rank == 0:
                log_dict = {"[Kodak] PSNR": psnrs.avg,
                            "[Kodak] MS-SSIM": ms_ssim_dbs.avg,
                            "[Kodak] BPP": bpps.avg,
                            "[Kodak] loss": losses.avg,
                            "[Kodak] Step": global_step,
                            }
                wandb.log(log_dict, step=global_step)
    return losses.avg


def parse_args(argv):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='./config/mt.yaml',
                        help='Path to config file to replace defaults.')
    parser.add_argument('--seed', type=int, default=1024,
                        help='Random seed.')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='GPU id to use.')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only (and do not run training).')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--local_rank', type=int,
                        help='Local rank for distributed training.')

    # logging
    parser.add_argument('--exp-name', type=str, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        help='Experiment name, unique id for trainers, logs.')
    parser.add_argument('--wandb', action="store_true",
                        help='Use wandb for logging.')
    parser.add_argument('--print-every', type=int, default=30,
                        help='Frequency of logging.')

    # dataset
    parser.add_argument('--dataset-path', default=['/media/Dataset/openimages/**/'],
                        help='Path to the dataset')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for dataloader.')
    parser.add_argument('--training-img-size', type=tuple, default=(256, 256),
                        help='Size of the training images.')
    parser.add_argument('--eval-dataset-path', type=str,
                        help='Path to the evaluation dataset')

    # optimization
    parser.add_argument('--distortion_metric', type=str, default='MSE',
                        help='Distortion type, MSE/SSIM/Perceptual.')
    parser.add_argument('--lambda_value', type=float, default=1,
                        help='Weight for the commitment loss.')

    # Optimizer configuration parameters
    parser.add_argument('--optimizer_type', type=str, default='AdamW',
                        help='The type of optimizer to use')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='The minimum learning rate for the learning rate policy')
    parser.add_argument('--min_lr', type=float, default=1e-4,
                        help='The minimum learning rate for the learning rate policy')
    parser.add_argument('--max_lr', type=float, default=1e-4,
                        help='The maximum learning rate for the learning rate policy')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98),
                        help='The beta values to use for the optimizer')
    parser.add_argument('--warmup_epoch', type=int, default=5,
                        help='The number of epochs to use for the warmup')
    parser.add_argument('--weight_decay', type=float, default=0.03,
                        help='The weight decay value for the optimizer')
    parser.add_argument('--clip-max-norm', type=float, default=1.0,
                        help='Gradient clipping for stable training.')

    # trainer
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run the training.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for the training.')
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint model")
    parser.add_argument('--save', action="store_true",
                        help="Save the model at every epoch (no overwrite).")
    parser.add_argument('--save-every', type=int, default=10000,
                        help='Frequency of saving the model.')
    parser.add_argument('--test-every', type=int, default=5000,
                        help='Frequency of running validation.')

    # model
    parser.add_argument('--net', type=str, default='MIT3D',
                        help='Model architecture.')
    args = parser.parse_args(argv)
    return args


def main(argv):
    global config
    config = parse_args(argv)

    global local_rank
    if config.multi_gpu:
        dist.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    # torch.backends.cudnn.benchmark = True

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    config.device = device
    job_type = 'test' if config.test_only else 'train'
    exp_name = config.net + " " + config.exp_name
    global workdir
    workdir, logger = logger_configuration(exp_name, job_type,
                                           method=config.net, save_log=(not config.test_only and local_rank == 0))
    net = MaskedImageModelingTransformer().to(device)
    if config.multi_gpu:
        net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)
    else:
        net = torch.nn.DataParallel(net)

    if config.wandb and local_rank == 0:
        print("=============== use wandb ==============")
        wandb_init_kwargs = {
            'project': 'MT',
            'name': exp_name,
            'save_code': True,
            'job_type': job_type,
            'config': config.__dict__
        }
        wandb.init(**wandb_init_kwargs)

        shutil.copy(config.config, join(workdir, 'config.yaml'))

    config.logger = logger
    logger.info(config.__dict__)

    if config.multi_gpu:
        train_dataset, test_dataset = get_dataset(config.dataset_path, config.eval_dataset_path)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True,
                                                   batch_size=config.batch_size,
                                                   worker_init_fn=worker_init_fn_seed,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=False)
    else:
        train_loader, test_loader = get_loader(config.dataset_path, config.eval_dataset_path,
                                               config.num_workers, config.batch_size)

    optimizer_cfg = {'lr': config.max_lr,
                     'betas': config.betas,
                     'weight_decay': config.weight_decay
                     }
    optimizer = getattr(torch.optim, config.optimizer_type)(net.parameters(), **optimizer_cfg)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.epochs * 0.9), gamma=0.1)

    global global_step
    global_step = 0

    if config.checkpoint is not None:
        load_weights(net, config.checkpoint, device)
    else:
        logger.info("No pretrained model is loaded.")

    if config.test_only:
        test(net, test_loader, device, logger)
    else:
        steps_epoch = global_step // train_loader.__len__()
        init_lambda_value = config.lambda_value
        for epoch in range(steps_epoch, config.epochs):
            if config.warmup:
                # for lambda warmup
                if epoch <= int(config.epochs * 0.1):
                    config.lambda_value = init_lambda_value * 10
                else:
                    config.lambda_value = init_lambda_value

            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            logger.info(f"Lambda value: {config.lambda_value}")
            train_one_epoch(epoch, net, train_loader, test_loader, optimizer, device, logger)
            lr_scheduler.step()


if __name__ == '__main__':
    main(sys.argv[1:])
