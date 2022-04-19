import math
import numpy as np
from tqdm import tqdm
from typing import Optional
from typing import Tuple
from collections import OrderedDict

import torch
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d

from models.base import BaseModule
from models.layers import *
from utils.ops import *

class TransformerEncoder(BaseModule):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, 
                 kernel_size=1, p_dropout=0.0, window_size=None, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, 
                                    n_heads, window_size=window_size, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, 
                                       filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x

class Diffusion(BaseModule):

    def __init__(self, hparams):
        super(Diffusion, self).__init__()
        self.hparams = hparams
        self.estimator = GradLogPEstimator2d(hparams.dec_dim, pe_scale=self.hparams.pe_scale, channels=self.hparams.dec_channels, n_feats=self.hparams.n_feats, use_sty=self.hparams.dec_use_sty, use_spk=self.hparams.dec_use_spk, sty_emb_dim=self.hparams.sty_embedding_dim, spk_emb_dim=self.hparams.spk_embedding_dim) 
        self.n_feats = hparams.n_feats
        self.beta_min = hparams.beta_min
        self.beta_max = hparams.beta_max
        self.pe_scale = hparams.pe_scale
        
    def get_noise(self, t, beta_init, beta_term, cumulative=False):
        if cumulative:
            noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
        else:
            noise = beta_init + (beta_term - beta_init)*t
        return noise

    def get_gamma(self, s, t, beta_init, beta_term):
        gamma = beta_init*(t-s) + 0.5*(beta_term-beta_init)*(t**2-s**2)
        gamma = torch.exp(-0.5*gamma)
        return gamma

    def get_mu(self, s, t):
        gamma_0_s = self.get_gamma(0, s, self.beta_min, self.beta_max)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        gamma_s_t = self.get_gamma(s, t, self.beta_min, self.beta_max)
        mu = gamma_s_t * ((1-gamma_0_s**2) / (1-gamma_0_t**2))
        return mu        

    def get_nu(self, s, t):
        gamma_0_s = self.get_gamma(0, s, self.beta_min, self.beta_max)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        gamma_s_t = self.get_gamma(s, t, self.beta_min, self.beta_max)
        nu = gamma_0_s * ((1-gamma_s_t**2) / (1-gamma_0_t**2))
        return nu

    def get_sigma(self, s, t):
        gamma_0_s = self.get_gamma(0, s, self.beta_min, self.beta_max)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        gamma_s_t = self.get_gamma(s, t, self.beta_min, self.beta_max)
        sigma = torch.sqrt(((1 - gamma_0_s**2) * (1 - gamma_s_t**2)) / (1 - gamma_0_t**2))
        return sigma        

    def get_kappa(self, t, h, noise):
        nu = self.get_nu(t-h, t)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        kappa = (nu*(1-gamma_0_t**2)/(gamma_0_t*noise*h) - 1)
        return kappa

    def get_omega(self, t, h, noise):
        mu = self.get_mu(t-h, t)
        kappa = self.get_kappa(t, h, noise)
        gamma_0_t = self.get_gamma(0, t, self.beta_min, self.beta_max)
        omega = (mu-1)/(noise*h) + (1+kappa)/(1-gamma_0_t**2) - 0.5
        return omega 


    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self.get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, sty=None, spk=None, frame_level_condition=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in tqdm(range(n_timesteps)):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self.get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mu, t, mask, spk=spk, sty=sty, frame_level_condition=frame_level_condition)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                es = self.estimator(xt, mu, t, mask, spk=spk, sty=sty, frame_level_condition=frame_level_condition)
                dxt = 0.5 * (mu - xt - es)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt


    def fast_maximum_likelihood_reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, sty=None, spk=None, frame_level_condition=None):
        print('Conduct fast maximum likelihood reverse diffusion ...')
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in tqdm(range(n_timesteps)):
            t = (1.0 - i*h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                 device=z.device)            
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = self.get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)

            kappa_t_h = self.get_kappa(t, h, noise_t) 
            omega_t_h = self.get_omega(t, h, noise_t)
            sigma_t_h = self.get_sigma(t-h, t)
 
            es = self.estimator(xt, mu, t, mask, spk=spk, sty=sty, frame_level_condition=frame_level_condition)

            dxt = ((0.5+omega_t_h)*(xt - mu) + (1+kappa_t_h) * es)
            dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                           requires_grad=False)
            dxt_stoc = dxt_stoc * sigma_t_h

            dxt = dxt * noise_t * h + dxt_stoc
            xt = (xt + dxt) * mask
        return xt
         
    @torch.no_grad()
    def inference(self, z, mask, mu, n_timesteps, stoc=False, sty=None, spk=None, frame_level_condition=None, fast_maximum_likelihood_sampling=False):
        if not fast_maximum_likelihood_sampling:
            return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, sty=sty, spk=spk, frame_level_condition=frame_level_condition)
        else:
            return self.fast_maximum_likelihood_reverse_diffusion(z, mask, mu, n_timesteps, stoc, sty=sty, spk=spk, frame_level_condition=frame_level_condition)

    def loss_t(self, x0, mask, mu, t, sty=None, spk=None, frame_level_condition=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self.get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mu, t, mask, sty=sty, spk=spk, frame_level_condition=frame_level_condition)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss

    def forward(self, x0, mask, mu, offset=1e-5, sty=None, spk=None, frame_level_condition=None, likelihood_weighting=False):
        if likelihood_weighting:
            t = self.sample_importance_weighted_time_for_likelihood(x0.shape[0], offset=offset)
            t = torch.from_numpy(t).to(x0)
        else:
            t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device, 
                           requires_grad=False)
            t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, sty=sty, spk=spk, frame_level_condition=frame_level_condition)
