import torch.optim as optim
import torch
import copy


class GradianceOptimizer(optim.Optimizer):
    def __init__(self, params, aggregated_unbiased_grads, lr=0.01, beta=0.99):
        super(GradianceOptimizer, self).__init__(params, {"lr": lr})
        self.beta = beta
        self.prior_grads = list(aggregated_unbiased_grads.values())

    def step(self):
        # Perform the optimization step
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    # perform gradiance step
                    p.grad.data = self.beta * self.prior_grads[idx] + p.grad.data
                    # p.grad.data = self.beta * self.prior_grads[idx] + (1 - self.beta) * p.grad.data
                    # p.grad.data = self.beta * self.prior_grads[param_name] + (1 - beta) * p.grad.data
                    
                    # update model weights
                    p.data -= group['lr'] * p.grad.data
                    # update the prior grads
                    self.prior_grads[idx] = copy.deepcopy(p.grad.data)