import logging
import math
import os

import numpy as np
import torch
import torchvision


def save_config(config, file_path):
    import json
    info_json = json.dumps(config, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(file_path, 'w')
    f.write(info_json)


def logger_configuration(filename, phase, method='', save_log=True):
    logger = logging.getLogger(" ")
    workdir = './history/{}/{}'.format(method, filename)
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    if save_log:
        makedirs(workdir)
        makedirs(samples)
        makedirs(models)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger


def single_plot(epoch, global_step, real, gen, config):
    images = [real, gen]
    filename = "{}/NTSCCModel_{}_epoch{}_step{}.png".format(config.samples, config.trainset, epoch, global_step)
    torchvision.utils.save_image(images, filename)


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, base_dir + "/checkpoint_best_loss.pth.tar")
    else:
        torch.save(state, base_dir + "/" + filename)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def bpp_snr_to_kdivn(bpp, SNR):
    snr = 10 ** (SNR / 10)
    kdivn = bpp / 3 / np.log2(1 + snr)
    return kdivn


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def BLN2BCHW(x, H, W):
    B, L, N = x.shape
    return x.reshape(B, H, W, N).permute(0, 3, 1, 2)


def BCHW2BLN(x):
    return x.flatten(2).permute(0, 2, 1)


def CalcuPSNR_int(img1, img2, max_val=255.):
    float_type = 'float64'
    img1 = np.round(torch.clamp(img1, 0, 1).detach().cpu().numpy() * 255)
    img2 = np.round(torch.clamp(img2, 0, 1).detach().cpu().numpy() * 255)

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def CalcuROIPSNR_int(img1, img2, mask, bar=0.1, max_val=255.):
    roi_region = torch.where(mask > bar)
    float_type = 'float64'
    img1 = np.round(torch.clamp(img1[roi_region], 0, 1).detach().cpu().numpy() * 255)
    img2 = np.round(torch.clamp(img2[roi_region], 0, 1).detach().cpu().numpy() * 255)

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def load_weights(net, model_path, device, remove_keys=None):
    try:
        pretrained = torch.load(model_path, map_location=device)['state_dict']
    except:
        pretrained = torch.load(model_path, map_location=device)
    result_dict = {}
    for key, weight in pretrained.items():
        result_key = key
        load_flag = True
        if 'attn_mask' in key:
            load_flag = False
        if remove_keys is not None:
            for remove_key in remove_keys:
                if remove_key in key:
                    load_flag = False
        if load_flag:
            result_dict[result_key] = weight
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained


def load_optim(optim, model_path, device):
    try:
        pretrained = torch.load(model_path, map_location=device)['optimizer']
    except:
        pretrained = torch.load(model_path, map_location=device)
    optim.load_state_dict(pretrained)
    del pretrained


def load_checkpoint(logger, device, global_step, net, optimizer_G, aux_optimizer, model_path):
    logger.info("Loading " + str(model_path))
    pre_dict = torch.load(model_path, map_location=device)

    global_step = pre_dict["global_step"]

    result_dict = {}
    for key, weight in pre_dict["state_dict"].items():
        result_key = key
        if 'mask' not in key:
            result_dict[result_key] = weight
    net.load_state_dict(result_dict, strict=False)

    # optimizer_G.load_state_dict(pre_dict["optimizer"])
    aux_optimizer.load_state_dict(pre_dict["aux_optimizer"])
    # lr_scheduler.load_state_dict(pre_dict["lr_scheduler"])

    return global_step


def load_pretrained(logger, device, net, model_path):
    logger.info("Loading " + str(model_path))
    pre_dict = torch.load(model_path, map_location=device)
    result_dict = {}
    for key, weight in pre_dict["state_dict"].items():
        result_key = key
        if 'mask' not in key:
            result_dict[result_key] = weight
    net.load_state_dict(result_dict, strict=False)


def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(math.log(max_val), math.log(min_val), num)
    else:
        values = np.linspace(math.log(min_val), math.log(max_val), num)
    values = np.exp(values)
    return values


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)
