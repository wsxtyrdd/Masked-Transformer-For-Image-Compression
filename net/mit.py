import math
import os
import time
from functools import lru_cache

import constriction
import numpy as np
import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional

from layer.layer_utils import quantize_ste, make_conv
from layer.transformer import TransformerBlock, SwinTransformerBlock
from net.elic import ELICAnalysis, ELICSynthesis


class GaussianMixtureEntropyModel(nn.Module):
    def __init__(
            self,
            minmax: int = 32
    ):
        super().__init__()
        self.minmax = minmax
        self.samples = torch.arange(-minmax, minmax + 1, 1, dtype=torch.float32).cuda()
        self.laplace = torch.distributions.Laplace(0, 1)
        self.pmf_laplace = self.laplace.cdf(self.samples + 0.5) - self.laplace.cdf(self.samples - 0.5)
        self.gaussian_conditional = GaussianConditional(None)

    def get_GMM_likelihood(self, latent_hat, params):
        B, C, H, W = latent_hat.size()
        probs, mean, scale = params.chunk(3, dim=1)
        scale = abs(scale).reshape(B, 3, C, H, W)  # B 3C H W
        probs = torch.softmax(probs.reshape(B, 3, C, H, W), dim=1)  # B 3C H W
        mean = mean.reshape(B, 3, C, H, W)  # B 3C H W
        likelihoods_0 = self.gaussian_conditional._likelihood(latent_hat, scale[:, 0], means=mean[:, 0])
        likelihoods_1 = self.gaussian_conditional._likelihood(latent_hat, scale[:, 1], means=mean[:, 1])
        likelihoods_2 = self.gaussian_conditional._likelihood(latent_hat, scale[:, 2], means=mean[:, 2])

        likelihoods_0 = self.gaussian_conditional.likelihood_lower_bound(likelihoods_0)
        likelihoods_1 = self.gaussian_conditional.likelihood_lower_bound(likelihoods_1)
        likelihoods_2 = self.gaussian_conditional.likelihood_lower_bound(likelihoods_2)

        likelihoods = 0.999 * (probs[:, 0] * likelihoods_0 + probs[:, 1] * likelihoods_1 + probs[:, 2] * likelihoods_2)
        + 0.001 * (self.laplace.cdf(latent_hat + 0.5) - self.laplace.cdf(latent_hat - 0.5))
        return likelihoods

    def get_GMM_pmf(self, probs, means, scales):
        L = self.samples.size(0)
        num_symbol = probs.size(1)

        # samples = self.samples.unsqueeze(0).unsqueeze(0).repeat(3, num_symbol, 1)  # 3 N 65
        # samples = samples - means.unsqueeze(-1)
        # upper = self._standardized_cumulative((0.5 - samples) / abs(scales).unsqueeze(-1))
        # lower = self._standardized_cumulative((-0.5 - samples) / abs(scales).unsqueeze(-1))
        # pmf_gmm = (probs.unsqueeze(-1) * (upper - lower)).sum(dim=0)
        # pmf = 0.999 * pmf_gmm + 0.001 * self.pmf_laplace
        # pmf_clip = pmf / pmf.sum(dim=1, keepdim=True)

        samples = self.samples.unsqueeze(0).repeat(num_symbol, 1)  # N 65
        scales = scales.unsqueeze(-1).repeat(1, 1, L)  # 3 N 65
        means = means.unsqueeze(-1).repeat(1, 1, L)  # 3 N 65
        probs = probs.unsqueeze(-1).repeat(1, 1, L)  # 3 N 65
        # print("GMM params", probs[:, 21, 0], scales[:, 21, 0], means[:, 21, 0])
        likelihoods_0 = self.gaussian_conditional._likelihood(samples, scales[0], means=means[0])
        # print("latent", samples[21, 33])
        # print("likelihoods_0", likelihoods_0[21, 33])
        likelihoods_1 = self.gaussian_conditional._likelihood(samples, scales[1], means=means[1])
        likelihoods_2 = self.gaussian_conditional._likelihood(samples, scales[2], means=means[2])
        pmf_clip = (0.999 * (probs[0] * likelihoods_0 + probs[1] * likelihoods_1 + probs[2] * likelihoods_2)
                    + 0.001 * self.pmf_laplace)
        # print("GMM pmf", pmf_clip[21, 33])
        return pmf_clip

    # def _standardized_cumulative(self, inputs):
    #     half = float(0.5)
    #     const = float(-(2 ** -0.5))
    #     # Using the complementary error function maximizes numerical precision.
    #     return half * torch.erfc(const * inputs)

    def compress(self, symbols, probs, means, scales):
        # t0 = time.time()
        # print("compress", t0)
        pmf_clip = self.get_GMM_pmf(probs, means, scales)

        # likelihoods = []
        # for i in range(symbols.size(0)):
        # likelihoods.append(pmf_clip[i, int(symbols[i] + self.minmax)].item())
        # print(likelihoods[21], int(symbols[21] + self.minmax))
        # print("ESTIMATE", -np.log2(np.array(likelihoods)).sum() / 512 / 768)

        model_family = constriction.stream.model.Categorical()  # note empty `()`
        probabilities = pmf_clip.cpu().numpy().astype(np.float64)
        symbols = symbols.reshape(-1)
        symbols = (symbols + self.minmax).cpu().numpy().astype(np.int32)
        # print(symbols.shape, probabilities.shape)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(symbols, model_family, probabilities)
        compressed = encoder.get_compressed()
        # print(encoder.num_bits() / 512 / 768)
        return compressed

    def decompress(self, compressed, probs, means, scales):
        pmf = self.get_GMM_pmf(probs, means, scales).cpu().numpy().astype(np.float64)
        model = constriction.stream.model.Categorical()
        decoder = constriction.stream.queue.RangeDecoder(compressed)
        symbols = decoder.decode(model, pmf)
        symbols = torch.from_numpy(symbols).to(probs.device) - self.minmax
        symbols = torch.tensor(symbols, dtype=torch.float32)
        return symbols


@lru_cache()
def get_coding_order(target_shape, context_mode, device, step=12):
    if context_mode == 'quincunx':
        context_tensor = torch.tensor([[4, 2, 4, 0], [3, 4, 3, 4], [4, 1, 4, 2]]).to(device)
    elif context_mode == 'checkerboard2':
        context_tensor = torch.tensor([[1, 0], [0, 1]]).to(device)
    elif context_mode == 'checkerboard4':
        context_tensor = torch.tensor([[0, 2], [3, 1]]).to(device)
    elif context_mode == 'qlds':
        B, C, H, W = target_shape

        def get_qlds(H, W):
            n, m, g = 0, 0, 1.32471795724474602596
            a1, a2 = 1.0 / g, 1.0 / g / g
            context_tensor = torch.zeros((H, W)).to(device) - 1
            while m < H * W:
                n += 1
                x = int(round(((0.5 + n * a1) % 1) * H)) % H
                y = int(round(((0.5 + n * a2) % 1) * W)) % W
                if context_tensor[x, y] == -1:
                    context_tensor[x, y] = m
                    m += 1
            return context_tensor

        context_tensor = torch.tensor(get_qlds(H, W), dtype=torch.int)

        def gamma_func(alpha=1.):
            return lambda r: r ** alpha

        ratio = 1. * (np.arange(step) + 1) / step
        gamma = gamma_func(alpha=2.2)
        L = H * W  # total number of tokens
        mask_ratio = np.clip(np.floor(L * gamma(ratio)), 0, L - 1)
        for i in range(step):
            context_tensor = torch.where((context_tensor <= mask_ratio[i]) * (context_tensor > i),
                                         torch.ones_like(context_tensor) * i, context_tensor)
        return context_tensor
    else:
        context_tensor = context_mode
    B, C, H, W = target_shape
    Hp, Wp = context_tensor.size()
    coding_order = torch.tile(context_tensor, (H // Hp + 1, W // Wp + 1))[:H, :W]
    return coding_order


class MaskedImageTransformer(nn.Module):
    def __init__(self, latent_dim, dim=768, depth=12, num_heads=12, window_size=24,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, transformer='swin'):
        super().__init__()
        self.dim = dim
        self.depth = depth
        if transformer == 'swin':
            window_size = 4
            num_heads = 8
            self.blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
                for i in range(depth)])
        self.delta = 5.0
        self.embedding_layer = nn.Linear(latent_dim, dim)
        # self.positional_encoding = LearnedPosition(window_size * window_size, dim)
        self.entropy_parameters = nn.Sequential(
            make_conv(dim, dim * 4, 1, 1),
            nn.GELU(),
            make_conv(dim * 4, dim * 4, 1, 1),
            nn.GELU(),
            make_conv(dim * 4, latent_dim * 9, 1, 1),
        )

        self.gmm_model = GaussianMixtureEntropyModel()
        self.laplace = torch.distributions.Laplace(0, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim), requires_grad=True)

    def forward_with_given_mask(self, latent_hat, mask, slice_size=None):
        B, C, H, W = latent_hat.size()
        input_resolution = (H, W)
        x = latent_hat.flatten(2).transpose(1, 2)  # B L N
        mask_BLN = mask.flatten(2).transpose(1, 2)  # B L N
        x_masked = x * mask_BLN + self.mask_token * (1 - mask_BLN)

        x_masked = self.embedding_layer(x_masked / self.delta)
        # x = self.positional_encoding(x)
        for _, blk in enumerate(self.blocks):
            x_masked = blk(x_masked, input_resolution, slice_size)
        x_out = x_masked.transpose(1, 2).reshape(B, self.dim, H, W)
        params = self.entropy_parameters(x_out)
        return params

    def forward_with_random_mask(self, latent):
        B, C, H, W = latent.size()
        latent_hat = quantize_ste(latent)

        def generate_random_mask(latent, r):
            mask_loc = torch.randn(H * W).to(latent.get_device())
            threshold = torch.sort(mask_loc)[0][r]
            mask = torch.where(mask_loc > threshold, torch.ones_like(mask_loc), torch.zeros_like(mask_loc))
            mask = mask.reshape(1, 1, H, W).repeat(B, C, 1, 1)
            return mask

        # r = math.floor(0.99 * H * W)
        r = math.floor(np.random.uniform(0.05, 0.99) * H * W)  # drop probability
        mask = generate_random_mask(latent_hat, r)
        # print(mask.sum(), mask.shape)  1=keep 0=drop
        params = self.forward_with_given_mask(latent_hat, mask)
        likelihoods = self.gmm_model.get_GMM_likelihood(latent_hat, params)
        likelihoods_masked = torch.where(mask == 1., torch.ones_like(likelihoods), likelihoods)
        return latent_hat, likelihoods_masked

    @torch.no_grad()
    def inference(self, latent, context_mode='qlds', slice_size=None):
        # B, C, H, W = latent.size()
        latent_hat = quantize_ste(latent)
        coding_order = get_coding_order(latent.shape, context_mode, latent_hat.get_device(), step=12)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(latent.shape[0], latent.shape[1], 1, 1)
        total_steps = int(coding_order.max() + 1)
        likelihoods = torch.zeros_like(latent_hat)
        for i in range(total_steps):
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            # params_locations = encoding_locations.repeat(1, 3, 1, 1).reshape(B, 3, C, H, W)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent), torch.zeros_like(latent))
            params_i = self.forward_with_given_mask(latent_hat, mask_i, slice_size)
            likelihoods_i = self.gmm_model.get_GMM_likelihood(latent_hat, params_i)
            likelihoods[encoding_locations] = likelihoods_i[encoding_locations]
            # print(-torch.log2(likelihoods_i[encoding_locations]).sum().item() / 512 / 768)
            # print(likelihoods_i[encoding_locations][21])
        return latent_hat, likelihoods

    def compress(self, latent, context_mode='qlds'):
        B, C, H, W = latent.size()
        latent_hat = torch.round(latent)
        coding_order = get_coding_order(latent.shape, context_mode, latent_hat.get_device(), step=12)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(latent.shape[0], latent.shape[1], 1, 1)
        total_steps = int(coding_order.max() + 1)
        t0 = time.time()
        strings = []
        for i in range(total_steps):
            # print('STEP', i)
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            # print(encoding_locations.sum())
            params_locations = encoding_locations.repeat(1, 3, 1, 1).reshape(B, 3, C, H, W)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent), torch.zeros_like(latent))

            params_i = self.forward_with_given_mask(latent_hat, mask_i)

            probs, mean, scale = params_i.chunk(3, dim=1)
            scales_i = abs(scale).reshape(B, 3, C, H, W)[params_locations].reshape(3, -1)  # B 3C H W
            probs_i = torch.softmax(probs.reshape(B, 3, C, H, W), dim=1)[params_locations].reshape(3, -1)  # B 3C H W
            means_i = mean.reshape(B, 3, C, H, W)[params_locations].reshape(3, -1)  # B 3C H W

            string_i = self.gmm_model.compress(latent_hat[encoding_locations], probs_i, means_i, scales_i)
            strings.append(string_i)
        print('compress', time.time() - t0)
        return strings

    def decompress(self, strings, latent_size, device, context_mode='qlds'):
        B, C, H, W = latent_size
        coding_order = get_coding_order(latent_size, context_mode, device, step=12)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(B, C, 1, 1)
        total_steps = int(coding_order.max() + 1)
        t0 = time.time()
        latent_hat = torch.zeros(latent_size).to(device)
        for i in range(total_steps):
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            params_locations = encoding_locations.repeat(1, 3, 1, 1).reshape(B, 3, C, H, W)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent_hat), torch.zeros_like(latent_hat))
            params_i = self.forward_with_given_mask(latent_hat, mask_i)
            probs, mean, scale = params_i.chunk(3, dim=1)
            scales_i = abs(scale).reshape(B, 3, C, H, W)[params_locations].reshape(3, -1)  # B 3C H W
            probs_i = torch.softmax(probs.reshape(B, 3, C, H, W), dim=1)[params_locations].reshape(3, -1)  # B 3C H W
            means_i = mean.reshape(B, 3, C, H, W)[params_locations].reshape(3, -1)  # B 3C H W
            symbols_i = self.gmm_model.decompress(strings[i], probs_i, means_i, scales_i)
            latent_hat[encoding_locations] = symbols_i
        print('decompress', time.time() - t0)
        return latent_hat

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}


class MaskedImageModelingTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.g_a = ELICAnalysis()
        self.g_s = ELICSynthesis()
        self.mim = MaskedImageTransformer(192)

    def forward(self, x):
        y = self.g_a(x)
        y_hat, likelihoods = self.mim.forward_with_random_mask(y)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": likelihoods,
        }

    def inference(self, x):
        y = self.g_a(x)
        y_hat, likelihoods = self.mim.inference(y)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": likelihoods,
        }

    def real_inference(self, x):
        num_pixels = x.size(2) * x.size(3)
        y = self.g_a(x)
        strings = self.mim.compress(y)
        y_hat = self.mim.decompress(strings, y.shape, x.get_device())
        x_hat = self.g_s(y_hat)
        bpp = sum([string.size * 32 for string in strings]) / num_pixels
        return {
            "x_hat": x_hat,
            "bpp": bpp,
        }


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from PIL import Image


    def read_image_to_torch(path):
        input_image = Image.open(path).convert('RGB')
        input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
        input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
        input_image = input_image.unsqueeze(0) / 255
        return input_image


    x = read_image_to_torch('/media/Dataset/kodak/kodim04.png').cuda()
    # print(get_coding_order((1, 1, 4, 24, 24), 'qlds3d', 'cpu', step=12))
    model = MaskedImageModelingTransformer().cuda()
    model = nn.DataParallel(model)
    state_dict = torch.load(
        r'/media/D/wangsixian/ResiComm/history/MIT/MIT 2023-08-21 11:47:54/checkpoint_best_loss.pth.tar')['state_dict']
    result_dict = {}
    for key, weight in state_dict.items():
        result_key = key
        if 'attn_mask' not in key:
            result_dict[result_key] = weight
    model.load_state_dict(result_dict, strict=False)
    model.eval()
    with torch.no_grad():
        results = model.module.inference(x)
        psnr = 10 * torch.log10(1 / torch.mean((x - results['x_hat']) ** 2)).item()
        bpp = -torch.sum(torch.log2(results['likelihoods'])).item() / (x.size(2) * x.size(3))
        print(psnr, bpp)

        results = model.module.real_inference(x)
        x_hat = results['x_hat']
        psnr = 10 * torch.log10(1 / torch.mean((x - x_hat) ** 2)).item()
        print(psnr)
        print(results['bpp'])
