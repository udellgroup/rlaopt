from typing import Callable
import time

import wandb


class Logger:
    def __init__(self, log_freq: int, log_in_wandb: bool, wandb_kwargs: dict):
        self.log_freq = log_freq
        self.log_in_wandb = log_in_wandb
        if self.log_in_wandb:
            wandb.init(**wandb_kwargs)

        self.start_time = time.time()
        self.iter_time = 0
        self.cum_time = 0

    def _reset_timer(self):
        self.start_time = time.time()

    def _update_cum_time(self):
        self.iter_time = time.time() - self.start_time
        self.cum_time += self.iter_time

    def _compute_log(self, i: int, compute_fn: Callable, *args, **kwargs):
        if i % self.log_freq != 0:
            return None
        else:
            self._update_cum_time()
            metrics = compute_fn(*args, **kwargs)

            log_dict = {"iter_time": self.iter_time, "cum_time": self.cum_time}
            log_dict["metrics"] = metrics

            if self.log_in_wandb:
                wandb.log(log_dict, step=i)

            self._reset_timer()

            return log_dict

    def _terminate(self):
        if self.log_in_wandb:
            wandb.finish()
