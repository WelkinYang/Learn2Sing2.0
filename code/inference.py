import matplotlib.pyplot as plt
import os
import shutil
import string
import math
import datetime as dt

import sys
import librosa
import numpy as np
import os
import glob
import json
import time
from scipy.io import wavfile
from utils.utils import *
from model import Generator
from text import *
from utils.audio import *

model_dir = sys.argv[1]
test_files_path = sys.argv[2]
output_dir = sys.argv[3]
spk_id = int(sys.argv[4])
sty_id = int(sys.argv[5])
n_timesteps = int(sys.argv[6])
device = sys.argv[7]
use_fast_maximum_likelihood_sampling = (sys.argv[8] == 'True')

os.makedirs(output_dir, exist_ok=True)

mels_dir = os.path.join(output_dir, 'mels')
os.makedirs(mels_dir, exist_ok=True)

hparams = get_hparams_from_dir(model_dir)
checkpoint_path = latest_checkpoint_path(model_dir)

model = Generator(hparams).to(device)

load_checkpoint(checkpoint_path, model)
_ = model.eval()

print(test_files_path)

avg_rtf = []
for label_name in os.listdir(test_files_path):
    print(label_name)
    label_path = os.path.join(test_files_path, label_name)

    with open(label_path, 'r') as f:
        label_lines = f.readlines()

        phos = []
        pitch_ids = []
        real_durations = []

        for line in label_lines:
            pho, pitch_id, real_duration = line.strip().split('\t')
            pho = learn2sing_pho_to_id[pho]
            pitch_id = ttsing_pitch_to_id[pitch_id]
            real_duration = math.ceil(float(real_duration) * hparams.sample_rate / hparams.hop_size)

            phos.append(pho)
            pitch_ids.append(pitch_id)
            real_durations.append(real_duration)

        phos = np.asarray(phos, np.int32)
        pitch_ids = np.asarray(pitch_ids, np.int32)
        real_durations = np.asarray(real_durations, np.int32)

        phos = torch.LongTensor(phos).unsqueeze(0).to(device)
        pitch_ids = torch.LongTensor(pitch_ids).unsqueeze(0).to(device)
        real_durations = torch.LongTensor(real_durations).unsqueeze(0).to(device)

    inputs = torch.stack([phos, pitch_ids, real_durations], dim=1)
    input_lengths = torch.tensor([inputs.shape[-1]]).to(device)
    spk_id = torch.tensor([spk_id]).to(device)
    sty_id = torch.tensor([sty_id]).to(device)

    with torch.no_grad():
        t = dt.datetime.now()
        predicted_mel, mu_mel = model.inference(inputs, input_lengths, n_timesteps, spk=spk_id, sty=sty_id, temperature=1.0, stoc=False, length_scale=1.0, use_fast_maximum_likelihood_sampling=use_fast_maximum_likelihood_sampling)
        t = (dt.datetime.now() - t).total_seconds()
        cur_rtf = (t * hparams.sample_rate / (predicted_mel.shape[-1] * hparams.hop_size))
        avg_rtf.append(cur_rtf)
        predicted_mel = predicted_mel.squeeze(0).cpu().numpy()
        mu_mel = mu_mel.squeeze(0).cpu().numpy()
        mel_output_file = os.path.join(mels_dir, os.path.splitext(label_name)[0] + '.npy')
        np.save(mel_output_file, predicted_mel)

print(f'avg rtf : {sum(avg_rtf)*1.0 / len(avg_rtf)}')
