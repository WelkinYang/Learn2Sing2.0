import os
import string
import random
import numpy as np
import math

from torch.utils.data import DataLoader
import torch

from utils.ops import fix_len_compatibility
from utils.audio import load_wav
from text import *

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, hparams, fileid_list_path):
        self.hparams = hparams
        self.fileid_list = self.get_fileid_list(fileid_list_path)
        random.seed(hparams.seed)
        random.shuffle(self.fileid_list)

    def get_fileid_list(self, fileid_list_path):
        fileid_list = []
        with open(fileid_list_path, 'r') as f:
            for line in f.readlines():
                fileid_list.append(line.strip())

        return fileid_list

    def __len__(self):
        return len(self.fileid_list)

class Learn2SingDataset(BaseDataset):

    def __init__(self, hparams, feature_dirs, fileid_list_path):
        BaseDataset.__init__(self, hparams, fileid_list_path)
        self.label_dir = feature_dirs[0]
        self.mel_dir = feature_dirs[1]
        self.avg_mel_dir = feature_dirs[2]
        self.utt2spk = self.get_utt2spk(feature_dirs[3])
        self.utt2sty = self.get_utt2sty(feature_dirs[4])

    def get_utt2spk(self, utt2spk_path):
        utt2spk = {}
        with open(utt2spk_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                utt_id, spk = line.strip().split('\t')
                utt2spk[utt_id] = spk

        return utt2spk

    def get_utt2sty(self, utt2sty_path):
        utt2sty = {}
        with open(utt2sty_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                utt_id, sty = line.strip().split('\t')
                utt2sty[utt_id] = sty

        return utt2sty

    def parse_label(self, label_path):
        with open(label_path, 'r') as f:
            label_lines = f.readlines()

            phos = []
            pitch_ids = []
            real_durations = []

            for line in label_lines:
                pho, pitch_id, real_duration = line.strip().split('\t')
                pho = learn2sing_pho_to_id[pho]
                pitch_id = ttsing_pitch_to_id[pitch_id]
                real_duration = math.ceil(float(real_duration) * self.hparams.sample_rate / self.hparams.hop_size)

                phos.append(pho)
                pitch_ids.append(pitch_id)
                real_durations.append(real_duration)

            phos = np.asarray(phos, np.int32)
            pitch_ids = np.asarray(pitch_ids, np.int32)
            real_durations = np.asarray(real_durations, np.int32)
            return  phos, pitch_ids, real_durations


    def __getitem__(self, index):
        pho, pitch_id, real_duration = \
            self.parse_label(os.path.join(self.label_dir, self.fileid_list[index] + '.lab'))
        mel = np.load(os.path.join(self.mel_dir, self.fileid_list[index] + '.npy'))
        avg_mel = np.load(os.path.join(self.avg_mel_dir, self.fileid_list[index] + '.npy'))
        spk = np.asarray([int(self.utt2spk[self.fileid_list[index]])], np.int32)
        sty = np.asarray([int(self.utt2sty[self.fileid_list[index]])], np.int32)

        return (torch.LongTensor(pho), torch.LongTensor(pitch_id), torch.LongTensor(real_duration), torch.LongTensor(spk), torch.LongTensor(sty), torch.FloatTensor(mel), torch.FloatTensor(avg_mel))

class Learn2SingCollate():

    def __init__(self, hparams):
        self.hparams = hparams
        self.mel_dim = self.hparams.acoustic_dim

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        pho_padded = torch.LongTensor(len(batch), max_input_len)
        pitch_padded = torch.LongTensor(len(batch), max_input_len)
        real_duration_padded = torch.LongTensor(len(batch), max_input_len)
        spk = torch.LongTensor(len(batch))
        sty = torch.LongTensor(len(batch))

        pho_padded.zero_()
        pitch_padded.zero_()
        real_duration_padded.zero_()
        spk.zero_()
        sty.zero_()

        duration_list = []
        note_lengths_list = []
        for i in range(len(ids_sorted_decreasing)):
            pho = batch[ids_sorted_decreasing[i]][0]
            pitch = batch[ids_sorted_decreasing[i]][1]
            real_duration = batch[ids_sorted_decreasing[i]][2]
            _spk = batch[ids_sorted_decreasing[i]][3]
            _sty = batch[ids_sorted_decreasing[i]][4]

            pho_padded[i, :pho.size(0)] = pho
            pitch_padded[i, :pitch.size(0)] = pitch
            real_duration_padded[i, :real_duration.size(0)] = real_duration
            spk[i] = _spk
            sty[i] = _sty
            duration_list.append(sum(real_duration))

        max_target_len = max(duration_list)

        mel_padded = torch.FloatTensor(len(batch), self.mel_dim, max_target_len)
        avg_mel_padded = torch.FloatTensor(len(batch), self.mel_dim, max_target_len)
        mel_padded.zero_()
        avg_mel_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][-2].transpose(0, 1)
            avg_mel = batch[ids_sorted_decreasing[i]][-1].transpose(0, 1)
            if mel.size(1) >= duration_list[i]:
                mel = mel[:, :int(duration_list[i].numpy())]
                avg_mel = avg_mel[:, :int(duration_list[i].numpy())]
            else:
                mel = torch.nn.functional.pad(mel, (0, duration_list[i]-mel.size(1)))
                avg_mel = torch.nn.functional.pad(avg_mel, (0, duration_list[i]-avg_mel.size(1)))

            mel_padded[i, :, :mel.size(1)] = mel
            avg_mel_padded[i, :, :mel.size(1)] = avg_mel
            output_lengths[i] = mel.size(1)

        inputs = torch.stack([pho_padded, pitch_padded, real_duration_padded], dim=1)
        return inputs, input_lengths, mel_padded, avg_mel_padded, output_lengths, spk, sty
