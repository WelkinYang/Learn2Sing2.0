## Learn2Sing 2.0: Diffusion and Mutual Information-Based Target Speaker SVS by Learning from Singing Teacher
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2203.16408)
[![GitHub Stars](https://img.shields.io/github/stars/Wendison/VQMIVC?style=social)](https://github.com/WelkinYang/Learn2Sing2.0)

Official implementation of Learn2Sing 2.0. For all details check out our paper which is accepted by Interspeech 2022 via [this](https://arxiv.org/abs/2203.16408) link.

**Authors**: Heyang Xue, Xinsheng Wang, Yongmao Zhang, Lei Xie, Pengcheng Zhu, Mengxiao Bi.

## Abstract

**Demo page** : [link](https://welkinyang.github.io/Learn2Sing2.0/).

Building a high-quality singing corpus for a person who is not good at singing is non-trivial, thus making it challenging to create a singing voice synthesizer for this person. Learn2Sing is dedicated to synthesizing the singing voice of a speaker without his or her singing data by learning from data recorded by others, i.e., the singing teacher. Inspired by the fact that pitch is the key style factor to distinguish singing from speaking voice, the proposed Learn2Sing 2.0 first generates the preliminary acoustic feature with averaged pitch value in the phone level, which allows the training of this process for different styles, i.e., speaking or singing, share same conditions except for the speaker information. Then, conditioned on the specific style, a diffusion decoder, which is accelerated by a fast sampling algorithm during the inference stage, is adopted to gradually restore the final acoustic feature. During the training, to avoid the information confusion of the speaker embedding and the style embedding, mutual information is employed to restrain the learning of speaker embedding and style embedding. Experiments show that the proposed approach is capable of synthesizing high-quality singing voice for the target speaker without singing data with 10 decoding steps.

## Training and inference:
*  Before you can use this implementation, you need to modify the following：

1. Replace the phoneset and pitchset in text/symbols.py with your own set

2. Provide the path to the data in config.json, the testdata folder contains example files to demonstrate the format

* Training

		bash run.sh
  
* Inference

		bash syn.sh outputs target_speaker_id 0 decoding_steps cuda True
		
## Acknowledgements:
* The diffusion decoder is adapted from [GradTTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS);
* Estimation of mutual information is modified from [VQMIVC](https://github.com/Wendison/VQMIVC/);
* Vadim Popov performed a code review of the fast sampling algorithm part.
