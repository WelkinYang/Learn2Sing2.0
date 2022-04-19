import numpy as np

import torch


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def compute_l2_loss(self, predicted_features, target_features, input_mask):
        l2_loss = torch.sum((predicted_features - target_features) ** 2) / torch.sum(torch.ones_like(predicted_features) * input_mask)
        return l2_loss

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

class BaseModel(BaseModule):
     def __init__(self, hparams):
         super(BaseModel, self).__init__()
         self.hparams = hparams
         self.loss = {}

         self._init_loss()

     def _init_loss(self):
         for loss_name in self.hparams.loss_list:
             self.loss[loss_name] = 0.0

     def set_loss_stats(self, loss_list):
         for i, loss_name in enumerate(self.hparams.loss_list):
             self.loss[loss_name] = loss_list[i]

         return self.loss
