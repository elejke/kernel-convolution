import torch

from .utils_kernels import LinearDCT


class PolynomialKernel(torch.nn.Module):
    def __init__(self, c=1.0, degree=2):
        super(PolynomialKernel, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.tensor(c), requires_grad=False)
        self.degree = torch.nn.parameter.Parameter(torch.tensor(degree), requires_grad=False)

    def forward(self, x, w):
        w = w.view(w.size(0), -1).t()
        out = (x.matmul(w) + self.c) ** self.degree
        return out


class GaussianKernel(torch.nn.Module):
    def __init__(self, gamma=0.5):
        super(GaussianKernel, self).__init__()
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(gamma), requires_grad=False)

    def forward(self, x, w):
        # Calculate L2-norm
        l2 = x.unsqueeze(3) - w.view(1, 1, -1, w.size(0))
        l2 = torch.sum(l2 ** 2, 2)

        out = torch.exp(-self.gamma * l2)
        return out


class SigmoidKernel(torch.nn.Module):
    def __init__(self):
        super(SigmoidKernel, self).__init__()

    def forward(self, x, w):
        w = w.view(w.size(0), -1).t()
        out = x.matmul(w).tanh()
        return out


class DCTKernel(torch.nn.Module):
    def __init__(self, in_features=50, norm=None, bias=False, cuda=False):
        super(DCTKernel, self).__init__()
        self.n = in_features
        self.norm = norm
        self.cuda = cuda
        self.bias = bias
        self.dct = LinearDCT(in_features=self.n, norm=self.norm, cuda=self.cuda, bias=self.bias).__call__

    def forward(self, x, w):
        # Calculate DCT-statistic for each patch
        out = self.dct(self.dct(x.reshape(-1, self.n, self.n)).transpose(1, 2)).transpose(1, 2).sum(dim=(1, 2))
        return out


class MaskedDCTKernel(torch.nn.Module):
    def __init__(self, in_features=50, norm=None, bias=False, cuda=False, masked_area=20):
        super(MaskedDCTKernel, self).__init__()
        self.n = in_features
        self.norm = norm
        self.cuda = cuda
        self.bias = bias
        self.mask = torch.ones((self.n, self.n))
        self.mask[:masked_area, :masked_area] = 0
        self.dct = LinearDCT(in_features=self.n, norm=self.norm, cuda=self.cuda, bias=self.bias).__call__

    def forward(self, x, w):
        # Calculate DCT-statistic for each patch
        out = (self.mask * self.dct(self.dct(x.reshape(-1, self.n,
                                                       self.n)).transpose(1, 2)).transpose(1, 2)).sum(dim=(1, 2))
        return out
