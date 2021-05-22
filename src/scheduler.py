class ScheduledOpt:
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
            self._rate = rate
            self.optimizer.step()

    def rate(self, step=None):
        # https://arxiv.org/pdf/1706.03762.pdf
        if step is None:
            step = self._step
        return self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup ** (-1.5)
        )
