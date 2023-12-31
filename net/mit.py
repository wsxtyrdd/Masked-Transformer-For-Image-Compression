import math
import os
import time
from functools import lru_cache

import constriction
import numpy as np
import torch
import torch.nn as nn

from layer.layer_utils import quantize_ste, make_conv
from layer.transformer import TransformerBlock, SwinTransformerBlock
from net.elic import ELICAnalysis, ELICSynthesis


# from compressai.entropy_models import GaussianConditional


class GaussianMixtureEntropyModel(nn.Module):
    def __init__(
            self,
            minmax: int = 64
    ):
        super().__init__()
        self.minmax = minmax
        self.samples = torch.arange(-minmax, minmax + 1, 1, dtype=torch.float32).cuda()
        self.laplace = torch.distributions.Laplace(0, 1)
        self.pmf_laplace = self.laplace.cdf(self.samples + 0.5) - self.laplace.cdf(self.samples - 0.5)
        # self.gaussian_conditional = GaussianConditional(None)

    def update_minmax(self, minmax):
        self.minmax = minmax
        self.samples = torch.arange(-minmax, minmax + 1, 1, dtype=torch.float32).cuda()
        self.pmf_laplace = self.laplace.cdf(self.samples + 0.5) - self.laplace.cdf(self.samples - 0.5)

    def get_GMM_likelihood(self, latent_hat, probs, means, scales):
        gaussian1 = torch.distributions.Normal(means[0], scales[0])
        gaussian2 = torch.distributions.Normal(means[1], scales[1])
        gaussian3 = torch.distributions.Normal(means[2], scales[2])
        likelihoods_0 = gaussian1.cdf(latent_hat + 0.5) - gaussian1.cdf(latent_hat - 0.5)
        likelihoods_1 = gaussian2.cdf(latent_hat + 0.5) - gaussian2.cdf(latent_hat - 0.5)
        likelihoods_2 = gaussian3.cdf(latent_hat + 0.5) - gaussian3.cdf(latent_hat - 0.5)

        likelihoods = 0.999 * (probs[0] * likelihoods_0 + probs[1] * likelihoods_1 + probs[2] * likelihoods_2)
        + 0.001 * (self.laplace.cdf(latent_hat + 0.5) - self.laplace.cdf(latent_hat - 0.5))
        likelihoods = likelihoods + 1e-10
        return likelihoods

    def get_GMM_pmf(self, probs, means, scales):
        L = self.samples.size(0)
        num_symbol = probs.size(1)
        samples = self.samples.unsqueeze(0).repeat(num_symbol, 1)  # N 65
        scales = scales.unsqueeze(-1).repeat(1, 1, L)
        means = means.unsqueeze(-1).repeat(1, 1, L)
        probs = probs.unsqueeze(-1).repeat(1, 1, L)
        likelihoods_0 = self._likelihood(samples, scales[0], means=means[0])
        likelihoods_1 = self._likelihood(samples, scales[1], means=means[1])
        likelihoods_2 = self._likelihood(samples, scales[2], means=means[2])
        pmf_clip = (0.999 * (probs[0] * likelihoods_0 + probs[1] * likelihoods_1 + probs[2] * likelihoods_2)
                    + 0.001 * self.pmf_laplace)
        return pmf_clip

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def compress(self, symbols, probs, means, scales):
        pmf_clip = self.get_GMM_pmf(probs, means, scales)
        model_family = constriction.stream.model.Categorical()  # note empty `()`
        probabilities = pmf_clip.cpu().numpy().astype(np.float64)
        symbols = symbols.reshape(-1)
        symbols = (symbols + self.minmax).cpu().numpy().astype(np.int32)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(symbols, model_family, probabilities)
        compressed = encoder.get_compressed()
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
        probs, means, scales = params.chunk(3, dim=1)
        probs = torch.softmax(probs.reshape(B, 3, C, H, W), dim=1).transpose(0, 1)
        means = means.reshape(B, 3, C, H, W).transpose(0, 1)
        scales = torch.abs(scales).reshape(B, 3, C, H, W).transpose(0, 1).clamp(1e-10, 1e10)
        return probs, means, scales

    def forward_with_random_mask(self, latent):
        B, C, H, W = latent.size()
        half = float(0.5)
        noise = torch.empty_like(latent).uniform_(-half, half)
        latent_noise = latent + noise
        latent_hat = quantize_ste(latent)

        def generate_random_mask(latent, r):
            mask_loc = torch.randn(H * W).to(latent.get_device())
            threshold = torch.sort(mask_loc)[0][r]
            mask = torch.where(mask_loc > threshold, torch.ones_like(mask_loc), torch.zeros_like(mask_loc))
            mask = mask.reshape(1, 1, H, W).repeat(B, C, 1, 1)
            return mask

        r = math.floor(np.random.uniform(0.05, 0.99) * H * W)  # drop probability
        mask = generate_random_mask(latent_hat, r)
        mask_params = mask.unsqueeze(0).repeat(3, 1, 1, 1, 1)
        probs, means, scales = self.forward_with_given_mask(latent_hat, mask)
        likelihoods_masked = torch.ones_like(latent_hat)
        likelihoods = self.gmm_model.get_GMM_likelihood(latent_noise[mask == 0],
                                                        probs[mask_params == 0].reshape(3, -1),
                                                        means[mask_params == 0].reshape(3, -1),
                                                        scales[mask_params == 0].reshape(3, -1))
        likelihoods_masked[mask == 0] = likelihoods
        return latent_hat, likelihoods_masked

    def inference(self, latent_hat, context_mode='qlds', slice_size=None):
        coding_order = get_coding_order(latent_hat.shape, context_mode, latent_hat.get_device(), step=12)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(latent_hat.shape[0], latent_hat.shape[1],
                                                                              1, 1)
        total_steps = int(coding_order.max() + 1)
        likelihoods = torch.zeros_like(latent_hat)
        for i in range(total_steps):
            ctx_locations = (coding_order < i)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent_hat), torch.zeros_like(latent_hat))
            probs_i, means_i, scales_i = self.forward_with_given_mask(latent_hat, mask_i, slice_size)
            encoding_locations = (coding_order == i)
            mask_params_i = encoding_locations.unsqueeze(0).repeat(3, 1, 1, 1, 1)
            likelihoods_i = self.gmm_model.get_GMM_likelihood(latent_hat[encoding_locations],
                                                              probs_i[mask_params_i].reshape(3, -1),
                                                              means_i[mask_params_i].reshape(3, -1),
                                                              scales_i[mask_params_i].reshape(3, -1))
            likelihoods[encoding_locations] = likelihoods_i
        return likelihoods

    def compress(self, latent, context_mode='qlds'):
        B, C, H, W = latent.size()
        latent_hat = torch.round(latent)
        self.gmm_model.update_minmax(int(latent_hat.max().item()))
        coding_order = get_coding_order(latent.shape, context_mode, latent_hat.get_device(), step=12)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(latent.shape[0], latent.shape[1], 1, 1)
        total_steps = int(coding_order.max() + 1)
        t0 = time.time()
        strings = []
        for i in range(total_steps):
            # print('STEP', i)
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            mask_params_i = encoding_locations.unsqueeze(0).repeat(3, 1, 1, 1, 1)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent), torch.zeros_like(latent))
            probs_i, means_i, scales_i = self.forward_with_given_mask(latent_hat, mask_i)
            string_i = self.gmm_model.compress(latent_hat[encoding_locations],
                                               probs_i[mask_params_i].reshape(3, -1),
                                               means_i[mask_params_i].reshape(3, -1),
                                               scales_i[mask_params_i].reshape(3, -1))
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
            mask_params_i = encoding_locations.unsqueeze(0).repeat(3, 1, 1, 1, 1)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent_hat), torch.zeros_like(latent_hat))
            probs_i, means_i, scales_i = self.forward_with_given_mask(latent_hat, mask_i)
            symbols_i = self.gmm_model.decompress(strings[i],
                                                  probs_i[mask_params_i].reshape(3, -1),
                                                  means_i[mask_params_i].reshape(3, -1),
                                                  scales_i[mask_params_i].reshape(3, -1))
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
        # TODO Patch-wise inference for off-the-shelf Transformers
        y = self.g_a(x)
        y_hat = torch.round(y)
        likelihoods = self.mim.inference(y_hat)
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
