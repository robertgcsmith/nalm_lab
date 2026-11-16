"""Implementation of the original NAC and NALU as first introduced 
in Trask et al. (2018), [arXiv:1808.00508v1].

Row-vector convention: inputs (B, I), parameter matrices (I, O).
Initial parameter tensors and NALU epsilon are supplied explicitly: the paper
does not prescribe a complete initialisation scheme or a numerical epsilon.
"""

import torch
from torch import nn

class NAC(nn.Module):
    """Bias free accumulator with W = tanh(W_hat) * sigmoid(M_hat)."""

    def __init__(self, W_hat: torch.Tensor, M_hat: torch.Tensor):
        super().__init__()
        if W_hat.ndim != 2 or W_hat.shape != M_hat.shape:
            raise ValueError("W_hat and M_hat must have the same (I, O) shape")
        self.W_hat = nn.Parameter(W_hat.detach().clone())
        self.M_hat = nn.Parameter(M_hat.detach().clone())

    def forward(self, x: torch.Tensor, *, diagnostics: bool = False):
        """Return a = x @ W, optionally with {'W': W}; tensors stay attached."""
        W = self.W_hat.tanh() * self.M_hat.sigmoid()
        a = x @ W
        return (a, {"W": W}) if diagnostics else a


class NALU(nn.Module):
    """Tied arithmetic weights and the original input-dependent, bias free gate.

    W_hat, M_hat and G have shape (I, O). eps is a required finite positive
    constant used only inside log(abs(x) + eps).
    """

    def __init__(
        self, W_hat: torch.Tensor, M_hat: torch.Tensor, G: torch.Tensor, *, eps: float
    ):
        super().__init__()
        if W_hat.ndim != 2 or not (W_hat.shape == M_hat.shape == G.shape):
            raise ValueError("W_hat, M_hat and G must have the same (I, O) shape")
        if not 0 < eps < float("inf"):
            raise ValueError("eps must be finite and positive")
        self.W_hat = nn.Parameter(W_hat.detach().clone())
        self.M_hat = nn.Parameter(M_hat.detach().clone())
        self.G = nn.Parameter(G.detach().clone())
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, *, diagnostics: bool = False):
        """Optionally return (y, {'W': W, 'a': a, 'm': m, 'g': g}).

        Diagnostics retain autograd connections and are not cached on the module.
        """
        W = self.W_hat.tanh() * self.M_hat.sigmoid()
        a = x @ W
        m = ((x.abs() + self.eps).log() @ W).exp()
        g = (x @ self.G).sigmoid()
        y = g * a + (1 - g) * m
        return (y, {"W": W, "a": a, "m": m, "g": g}) if diagnostics else y
