import numpy as np

class Optimizer(object):
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self,
                 optimizer,
                 init_lr,
                 current_step=0,
                 warmup_steps=50000,
                 decay_learning_rate=0.5):

        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr = init_lr
        self.current_steps = current_step
        self.warmup_steps = warmup_steps
        self.decay_learning_rate = decay_learning_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()

    def synchronize(self):
        self.optimizer.synchronize()

    def skip_synchronize(self):
        self.optimizer.skip_synchronize()

    def get_lr_scale(self):
        if self.current_steps >= self.warmup_steps:
           lr_scale = np.power(self.decay_learning_rate, self.current_steps / self.warmup_steps)
        else:
           lr_scale = 1

        return lr_scale

    def update_learning_rate(self):
        self.current_steps += 1
        lr = self.init_lr * self.get_lr_scale()
        lr = np.maximum(1e-6, lr)
        self.lr = lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_lr(self):
        return self.lr

    def load_state_dict(self, d):
        self.optmizer.load_state_dict(d)

    def state_dict(self):
        return self.optimizer.state_dict()

    def param_groups(self):
        return self.optimizer.param_groups()
