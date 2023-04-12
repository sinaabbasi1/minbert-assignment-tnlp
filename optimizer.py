from typing import Callable, Iterable, Tuple

import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.t = 0

    def step(self, closure: Callable = None):
        self.t += 1
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                if len(state) == 0:
                    state['m'] = 0.0
                    state['v'] = 0.0

                m = state['m']
                v = state['v']

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]
                eps = group["eps"]

                # Update first and second moments of the gradients
                m = (beta1 * m) + ((1.0 - beta1) * grad)
                v = (beta2 * v) + ((1.0 - beta2) * torch.square(grad))

                state['m'] = m
                state['v'] = v

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                
                if correct_bias:
                    m_hat = m / (1.0 - beta1 ** self.t)
                    v_hat = v / (1.0 - beta2 ** self.t)

                # Update parameters
                p.data = p.data - ((alpha * m_hat) / (torch.sqrt(v_hat) + eps)) 
                p.data = p.data - (alpha * weight_decay * p.data)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss
