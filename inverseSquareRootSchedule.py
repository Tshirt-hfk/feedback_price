

class InverseSquareRootSchedule:
    """Optim wrapper that implements rate."""

    def __init__(self, optimizer, lr, warmup_updates, warmup_init_lr=1e-6):
        self.step_num = 0
        self._lr = 0
        self._optimizer = optimizer
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.lr_step = (lr - warmup_init_lr) / warmup_updates
        self.decay_factor = lr * warmup_updates**0.5

    def get_lr(self):
        return self._lr

    def step(self):
        """Update parameters and rate"""
        self.step_num += 1
        if self.step_num < self.warmup_updates:
            self._lr = self.warmup_init_lr + self.step_num*self.lr_step
        else:
            self._lr = self.decay_factor * self.step_num**-0.5
        for p in self._optimizer.param_groups:
            p['lr'] = self._lr
