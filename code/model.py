import sys
import numpy as np
import math
import random
sys.path.append("../../")

import torch
from torch import nn
from torch.nn import functional as F

from models.base import *
from models.modules import *
from models.layers import *
from utils.ops import *
from text.symbols import *
from text import *

class TextEncoder(BaseModule):
    def __init__(self, hparams):
        super(TextEncoder, self).__init__()
        self.n_channels = hparams.phone_embedding_channels + hparams.pitch_embedding_channels
        self.enc_output_dim = hparams.enc_output_dim
        self.enc_filter_channels = hparams.enc_filter_channels
        self.enc_n_heads = hparams.enc_n_heads
        self.enc_n_layers = hparams.enc_n_layers
        self.enc_kernel_size = hparams.enc_kernel_size
        self.enc_p_dropout = hparams.enc_p_dropout
        self.enc_window_size = hparams.enc_window_size
        self.n_spks = hparams.n_spks

        self.phone_embedding = torch.nn.Embedding(len(learn2sing_phone_set), hparams.phone_embedding_channels)
        torch.nn.init.normal_(self.phone_embedding.weight, 0.0, hparams.phone_embedding_channels**-0.5)
        self.pitch_embedding = torch.nn.Embedding(len(ttsing_pitch_set), hparams.pitch_embedding_channels)
        torch.nn.init.normal_(self.pitch_embedding.weight, 0.0, hparams.pitch_embedding_channels**-0.5)

        self.prenet = ConvReluNorm(self.n_channels, self.n_channels, self.n_channels,
                                   kernel_size=5, n_layers=3, p_dropout=0.5)

        self.encoder = TransformerEncoder(self.n_channels + hparams.spk_embedding_dim, self.enc_filter_channels, self.enc_n_heads, self.enc_n_layers,
                               self.enc_kernel_size, self.enc_p_dropout, window_size=self.enc_window_size)

        self.proj_m = torch.nn.Conv1d(self.n_channels + hparams.spk_embedding_dim, self.enc_output_dim, 1)

        self.num_params()

    def forward(self, pho, pitch, spk, input_lengths):
        input_mask = torch.unsqueeze(sequence_mask(input_lengths, pho.size(1)), 1).to(pho.dtype)

        pho = self.phone_embedding(pho).transpose(1, 2)
        pitch = self.pitch_embedding(pitch).transpose(1, 2)
        encoder_inputs = torch.cat([pho, pitch], dim=1)

        h = self.prenet(encoder_inputs, input_mask)
        h = torch.cat([h, spk.unsqueeze(-1).repeat(1, 1, h.shape[-1])], dim=1)
        h = self.encoder(h, input_mask)

        z_mu = self.proj_m(h) * input_mask
        return z_mu, input_mask

class MutualInformationNetworks(BaseModel):

    def __init__(self, hparams):
        BaseModel.__init__(self, hparams)
        self.ss_mi_net = CLUBSample_reshape(hparams.spk_embedding_dim, hparams.sty_embedding_dim, hparams.mi_net_hidden_size)
        self.ps_mi_net = CLUBSample_reshape(hparams.spk_embedding_dim, hparams.pitch_enc_output_dim, hparams.mi_net_hidden_size)

    def loglikeli(self, content_emb, sty_emb, spk_emb, input_mask):
        spk_emb = spk_emb.unsqueeze(-1).repeat(1, 1, content_emb.shape[-1])
        sty_emb = sty_emb.unsqueeze(-1).repeat(1, 1, content_emb.shape[-1])
        sty_emb, spk_emb, input_mask = sty_emb.transpose(1, 2), spk_emb.transpose(1, 2), input_mask.transpose(1, 2)
        lld_ss_loss = -self.ss_mi_net.loglikeli(spk_emb, sty_emb, input_mask)
        return lld_ss_loss

    def mi_est(self, content_emb, sty_emb, spk_emb, input_mask):
        spk_emb = spk_emb.unsqueeze(-1).repeat(1, 1, content_emb.shape[-1])
        sty_emb = sty_emb.unsqueeze(-1).repeat(1, 1, content_emb.shape[-1])
        sty_emb, spk_emb, input_mask = sty_emb.transpose(1, 2), spk_emb.transpose(1, 2), input_mask.transpose(1, 2)

        mi_ss_loss = self.ss_mi_net.mi_est(spk_emb, sty_emb, input_mask)
        return mi_ss_loss
   
        
class Generator(BaseModel):


    def __init__(self, hparams):
        BaseModel.__init__(self, hparams)
        self.encoder = TextEncoder(hparams)
        self.length_regulator = LengthRegulator()
        self.decoder = Diffusion(hparams)

        self.spk_emb = torch.nn.Embedding(hparams.n_spks, hparams.spk_embedding_dim)
        self.sty_emb = torch.nn.Embedding(hparams.n_stys, hparams.sty_embedding_dim)

        self.num_params()

    def compute_prior_loss(self, mu_mels, target_mels, target_mask):
        prior_loss = torch.sum(0.5 * ((target_mels - mu_mels) ** 2 + math.log(2 * math.pi)) * target_mask)
        prior_loss = prior_loss / (torch.sum(target_mask) * self.hparams.n_feats)
        return prior_loss

    def get_content_pitch_spk(self, inputs, input_lengths, spk, sty):
        pho, pitch, real_duration = torch.split(inputs, split_size_or_sections=1, dim=1)
        pho, pitch, real_duration = pho.squeeze(1), pitch.squeeze(1), real_duration.squeeze(1)

        spk_emb = self.spk_emb(spk)
        sty_emb = self.sty_emb(sty)

        encoder_outputs, input_mask = self.encoder(pho, pitch, spk_emb, input_lengths)

        return encoder_outputs, spk_emb, sty_emb, real_duration, input_mask
        

    def forward(self, inputs, input_lengths, target_mels, avg_mels, target_lengths, spk, sty):
        encoder_outputs, spk, sty, real_duration, input_mask = self.get_content_pitch_spk(inputs, input_lengths, spk, sty)

        mu_mels = self.length_regulator(encoder_outputs.transpose(1, 2), real_duration).transpose(1, 2)
        target_mask = sequence_mask(target_lengths, mu_mels.size(-1)).unsqueeze(1).to(input_mask)
        mel_loss = self.compute_prior_loss(mu_mels, avg_mels, target_mask)

        if self.hparams.memory_efficient_training:
            max_offset = (target_lengths - self.hparams.efficient_len).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(target_lengths)

            target_mel_cut = torch.zeros(target_mels.shape[0], self.hparams.acoustic_dim, self.hparams.efficient_len, dtype=target_mels.dtype, device=target_mels.device)
            mu_mel_cut = torch.zeros(mu_mels.shape[0], self.hparams.n_feats, self.hparams.efficient_len, dtype=mu_mels.dtype, device=mu_mels.device)

            target_mel_cut_lengths = []
            for i, (target_mel_, out_offset_) in enumerate(zip(target_mels, out_offset)):
                target_mel_cut_length = self.hparams.efficient_len + (target_lengths[i] - self.hparams.efficient_len).clamp(None, 0)
                target_mel_cut_lengths.append(target_mel_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + target_mel_cut_length
                target_mel_cut[i, :, :target_mel_cut_length] = target_mel_[:, cut_lower:cut_upper]
                mu_mel_cut[i, :, :target_mel_cut_length] = mu_mels[i, :, cut_lower:cut_upper]

            target_mel_cut_lengths = torch.LongTensor(target_mel_cut_lengths)
            target_mel_cut_mask = sequence_mask(target_mel_cut_lengths).unsqueeze(1).to(target_mask)
            diff_loss = self.decoder(target_mel_cut, target_mel_cut_mask, mu_mel_cut, sty=sty)
        else:
            target_mels = F.pad(target_mels, (0, math.ceil(target_mels.size(-1)/4)*4-target_mels.size(-1)), "constant", 0)
            mu_mels = F.pad(mu_mels, (0, math.ceil(mu_mels.size(-1)/4)*4-mu_mels.size(-1)), "constant", 0)
            target_mask = sequence_mask(target_lengths, target_mels.size(-1)).unsqueeze(1).to(input_mask)
            diff_loss = self.decoder(target_mels, target_mask, mu_mels, sty=sty)

        loss_dict = self.set_loss_stats([mel_loss, diff_loss]) 

        return loss_dict, sty, encoder_outputs, spk, input_mask

    @torch.no_grad()
    def inference(self, inputs, input_lengths, n_timesteps, spk, sty, temperature=1.0, stoc=False, length_scale=1.0, use_fast_maximum_likelihood_sampling=False):
        
        pho, pitch, real_duration = torch.split(inputs, split_size_or_sections=1, dim=1)
        pho, pitch, real_duration = pho.squeeze(1), pitch.squeeze(1), real_duration.squeeze(1)
        spk = self.spk_emb(spk)
        sty = self.sty_emb(sty)

        encoder_outputs, input_mask = self.encoder(pho, pitch, spk, input_lengths)

        mu_mel = self.length_regulator(encoder_outputs.transpose(1, 2), real_duration).transpose(1, 2)

        mel_len = mu_mel.size(-1)
        mel_max_len = fix_len_compatibility(mu_mel.size(-1))
        padded_mu_mel = F.pad(mu_mel, (0, mel_max_len-mel_len), "constant", 0)
        sampling_mel = padded_mu_mel + torch.randn_like(padded_mu_mel, device=mu_mel.device) / temperature

        target_mask = sequence_mask(torch.tensor([padded_mu_mel.size(-1)]).to(padded_mu_mel.device), padded_mu_mel.size(-1)).unsqueeze(1).to(input_mask.dtype)
        predicted_mel = self.decoder.inference(sampling_mel, target_mask, padded_mu_mel, n_timesteps, stoc, sty=sty, fast_maximum_likelihood_sampling=use_fast_maximum_likelihood_sampling)[:, :, :mel_len]

        return predicted_mel, padded_mu_mel
