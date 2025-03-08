import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_dim=1, beta=0.999, epsilon=1e-5):
        super(ValueNorm, self).__init__()

        self.input_dim = input_dim
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon) # [input_dim]
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon) # [input_dim]
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2) # [input_dim]
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor) -> None:
        assert len(input_vector.shape) == 2
        assert input_vector.shape[-1] == self.input_dim

        batch_mean = input_vector.mean(dim=0)
        batch_sq_mean = (input_vector ** 2).mean(dim=0)

        weight = self.beta
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - weight))

    @torch.no_grad()
    def normalize(self, input_vector: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        mean = mean.unsqueeze(0)
        var = var.unsqueeze(0)

        out = (input_vector - mean) / torch.sqrt(var)

        return out

    def __call__(self, input_vector: torch.Tensor) -> torch.Tensor:
        return self.normalize(input_vector)

    @torch.no_grad()
    def denormalize(self, input_vector: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        mean = mean.unsqueeze(0)
        var = var.unsqueeze(0)

        out = input_vector * torch.sqrt(var) + mean

        return out
