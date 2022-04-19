import os
import sys
import json
import argparse
import time
import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from utils import utils, optimizer, ops
from dataset.dataset import Learn2SingDataset, Learn2SingCollate
import model

epoch = 1
global_step = 0

def train(hparams):

    global global_step
    global epoch

    logger = utils.get_logger(hparams.model_dir)
    logger.info(hparams)
    writer = SummaryWriter(log_dir=os.path.join(hparams.model_dir, "train"))

    torch.manual_seed(hparams.seed)

    train_dataset = Learn2SingDataset(hparams, hparams.train_feature_dirs, hparams.train_fileid_list_path)
    train_collate = Learn2SingCollate(hparams)

    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True,
                                   batch_size=hparams.batch_size, pin_memory=False,
                                   drop_last=True, collate_fn=train_collate)

    generator = model.Generator(hparams).cuda()
    g_parameters = list(generator.parameters())
    g_optimizer = optim.Adam(g_parameters, lr=hparams.g_learning_rate, betas=(hparams.betas[0], hparams.betas[1]))

    mi_net = model.MutualInformationNetworks(hparams).cuda()
    ss_mi_net_parameters = list(mi_net.ss_mi_net.parameters())
    ps_mi_net_parameters = list(mi_net.ps_mi_net.parameters()) 

    ss_mi_net_optimizer = optim.Adam(ss_mi_net_parameters, lr=hparams.mi_net_learning_rate, betas=(hparams.betas[0], hparams.betas[1]))
    ps_mi_net_optimizer = optim.Adam(ps_mi_net_parameters, lr=hparams.mi_net_learning_rate, betas=(hparams.betas[0], hparams.betas[1]))
    checkpoint_path = utils.latest_checkpoint_path(hparams.model_dir, "M_*.pth")
    if os.path.isfile(checkpoint_path):
        epoch, global_step = utils.load_mi_net_checkpoint_v1(checkpoint_path, generator, g_optimizer, mi_net, ss_mi_net_optimizer, ps_mi_net_optimizer)
    
    customer_g_optimizer = optimizer.Optimizer(g_optimizer, hparams.g_learning_rate,
                       global_step, hparams.warmup_steps, hparams.decay_learning_rate)

    customer_ss_mi_net_optimizer = optimizer.Optimizer(ss_mi_net_optimizer, hparams.mi_net_learning_rate,
                       global_step, hparams.mi_warmup_steps, hparams.decay_learning_rate)

    customer_ps_mi_net_optimizer = optimizer.Optimizer(ps_mi_net_optimizer, hparams.mi_net_learning_rate,
                       global_step, hparams.mi_warmup_steps, hparams.decay_learning_rate)

    for epoch in range(epoch, hparams.epochs + 1):
        train_one_mi_epoch(epoch, hparams, generator, customer_g_optimizer, mi_net, customer_ss_mi_net_optimizer, customer_ps_mi_net_optimizer, train_loader, logger, writer)

        if epoch % hparams.checkpoint_interval == 0:
            utils.save_mi_net_v1_checkpoint(generator, customer_g_optimizer, mi_net, customer_ss_mi_net_optimizer, customer_ps_mi_net_optimizer, customer_g_optimizer.get_lr(), epoch, global_step, os.path.join(hparams.model_dir, "M_{}.pth".format(epoch)))



def train_one_mi_epoch(epoch, hparams, generator, optimizer_g, mi_net, optimizer_ss_mi_net, optimizer_ps_mi_net, data_loader, logger, writer):
    global global_step
    
    generator.train()
    mi_net.train()
    for batch_idx, (inputs, input_lengths, target_mels, avg_mels, target_lengths, spk, sty) in enumerate(data_loader):
        inputs, input_lengths = inputs.cuda(non_blocking=True), input_lengths.cuda(non_blocking=True)
        target_mels, avg_mels, target_lengths = target_mels.cuda(non_blocking=True), avg_mels.cuda(non_blocking=True), target_lengths.cuda(non_blocking=True)
        spk, sty = spk.cuda(non_blocking=True), sty.cuda(non_blocking=True)

        loss_ss_mi_lld = 0
        loss_ps_mi_lld = 0

        for mi_iter_index in range(hparams.mi_iters): 
            optimizer_ss_mi_net.zero_grad()
            optimizer_ps_mi_net.zero_grad()

            content_emb, spk_emb, sty_emb, _, input_mask = generator.get_content_pitch_spk(inputs, input_lengths, spk, sty)
            loss_ss_mi_lld = mi_net.loglikeli(content_emb.detach(), sty_emb.detach(), spk_emb.detach(), input_mask)

            loss_ss_mi_lld.backward()

            ss_mi_grad_norm = ops.clip_grad_value_(mi_net.ss_mi_net.parameters(), 1)

            optimizer_ss_mi_net.step_and_update_lr()

        optimizer_g.zero_grad()
        loss_g_dict, sty_emb, content_emb, spk_emb, input_mask = generator(inputs, input_lengths, target_mels, avg_mels, target_lengths, spk, sty)

        mi_ss_loss = mi_net.mi_est(content_emb, sty_emb, spk_emb, input_mask)

        loss_g = sum(loss_g_dict.values()) + mi_ss_loss * 0.01 
        loss_g.backward()
        grad_norm = ops.clip_grad_value_(generator.parameters(), 5)
        optimizer_g.step_and_update_lr()

        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(inputs), len(data_loader.dataset),
            100. * batch_idx / len(data_loader),
            loss_g.item()))
        logger.info([f'{key}: {loss_g_dict[key].item()} ' for key in loss_g_dict.keys()]  + [f'mi_ss_loss: {mi_ss_loss} ' + f'loss_ss_mi_lld: {loss_ss_mi_lld} '] + [global_step, optimizer_g.get_lr()])

        global_step += 1

        if global_step % hparams.writer_interval == 0:
            predicted_mel, _ = generator.inference(inputs[:1], input_lengths[:1], 100, spk[:1], sty[:1], use_real_duration=True)
            #Write your features, below is an example
            scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
            scalar_dict.update({"loss/g/{}".format(key): loss_g_dict[key].item() for key in loss_g_dict.keys()})
            utils.summarize(
                writer=writer,
                global_step=global_step,
                images={"gt_mel": utils.plot_spectrogram_to_numpy(target_mels[0].data.cpu().numpy()), 
                        "p_mel": utils.plot_spectrogram_to_numpy(predicted_mel[0].data.cpu().numpy()), 
                },
                scalars=scalar_dict)

    logger.info('====> Epoch: {}'.format(epoch))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Json file for configuration')
    parser.add_argument('-l', '--logdir', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')

    args = parser.parse_args()
    hparams = utils.train_setup(args.config, args.logdir, args.model)
   
    train(hparams)

if __name__ == "__main__":
    main()
