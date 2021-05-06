
import torch
import torch.nn as nn
import numpy as np

from scipy import linalg

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init


# -----------------------------------------------------------------------------------------
class Permutation(nn.Module):
    
    def __init__(self, permutation: str, n_channel: int, u_channel: int=None):
        super(Permutation, self).__init__()

        assert permutation in ['reverse', 'random', 'inv1x1', 'cond_inv1x1']
        if permutation == 'inv1x1':
            self.permutater = InvertibleConv1x1(n_channel)
            # self.permutater = OneByOneConv(n_channel)
        elif permutation == 'cond_inv1x1':
            self.permutater = CondInvertibleConv1x1(n_channel, u_channel)
        else:
            self.permutater = ShufflePermutation(permutation, n_channel)
    
    def forward(self, x: Tensor, p=None):
        return self.permutater(x, p)
    
    def inverse(self, z: Tensor, p=None):
        return self.permutater.inverse(z, p)


# -----------------------------------------------------------------------------------------
class ShufflePermutation(nn.Module):

    def __init__(self, permutation: str, n_channel: int):
        super(ShufflePermutation, self).__init__()

        if permutation == 'reverse':
            self.direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.long)
            self.inverse_idx = ShufflePermutation.get_reverse(self.direct_idx, n_channel)
        if permutation == 'random':
            self.direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.long)
            np.random.shuffle(self.direct_idx)
            self.inverse_idx = ShufflePermutation.get_reverse(self.direct_idx, n_channel)
 
    def forward(self, x: Tensor, _):
        log_det_J = torch.tensor([0.0], device=x.device)
        return x[:, self.direct_idx, :], log_det_J

    def inverse(self, z: Tensor, _):
        return z[:, self.inverse_idx, :]

    @staticmethod
    def get_reverse(idx, n_channel: int):
        indices_inverse = np.zeros((n_channel,), dtype=np.long)
        for i in range(n_channel):
            indices_inverse[idx[i]] = i
        return indices_inverse


# -----------------------------------------------------------------------------------------
class InvertibleConv1x1(nn.Module):

    def __init__(self, channel: int):
        super(InvertibleConv1x1, self).__init__()

        self.W = nn.Parameter(torch.empty((channel, channel)))  # [channel,]
        init.normal_(self.W, std=0.01)

    def forward(self, x: Tensor, _):
        """
        x: [B, C, N]
        """
        z = torch.einsum('ij,bjn->bin', self.W, x)
        logdet = torch.slogdet(self.W)[1] * x.shape[-1]
        return z, logdet

    def inverse(self, z: Tensor, _):
        inv_W = torch.inverse(self.W)
        x = torch.einsum('ij,bjn->bin', inv_W, z)
        return x


# -----------------------------------------------------------------------------------------
class CondInvertibleConv1x1(nn.Module):

    def __init__(self, channel: int, uchannel: int):
        super(CondInvertibleConv1x1, self).__init__()

        self.w_channel = channel
        self.linear1 = nn.Linear(uchannel, 64)
        self.linear2 = nn.Linear(64, channel * channel)

    def forward(self, x: Tensor, c: Tensor):

        W = self.get_W(c)
        z = torch.einsum('bij,bjn->bin', W, x)
        logdet = torch.slogdet(W)[1] * x.shape[-1]
        return z, logdet

    def inverse(self, z: Tensor, c: Tensor):
        inv_W = torch.inverse(self.get_W(c))
        x = torch.einsum('bij,bjn->bin', inv_W, z)
        return x
    
    def get_W(self, c: Tensor):
        _c = self.linear1(c.transpose(1, 2))  # [B, N, 64]
        _c, _ = torch.max(_c, dim=1)  # [B, 64]
        return self.linear2(_c).view(-1, self.w_channel, self.w_channel)  # [B, C, C]



class OneByOneConv(nn.Module):
    """
    Invertible 1x1 convolution. From https://github.com/tonyduan/normalizing-flows
    [Kingma and Dhariwal, 2018.]
    """
    def __init__(self, dim):
        super(OneByOneConv, self).__init__()
        self.dim = dim
        W, _ = linalg.qr(np.random.randn(dim, dim))
        P, L, U = linalg.lu(W)
        self.register_buffer('P', torch.tensor(P, dtype=torch.float))
        # self.P = nn.Parameter(torch.tensor(P, dtype=torch.float), requires_grad=False)
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float))
        self.S = nn.Parameter(torch.tensor(np.diag(U).copy(), dtype=torch.float))
        self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype=torch.float), diagonal=1))
        self.W_inv = None

    def forward(self, x: Tensor):
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim)).to(x.device)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))

        # z = x @ self.P @ L @ (U + torch.diag(self.S))
        # z = (x.transpose(1, 2) @ W).transpose(1, 2)
        z = torch.einsum('ij,bin->bjn', W, x)

        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z: Tensor):
        if self.W_inv is None:
            L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim)).to(z.device)
            U = torch.triu(self.U, diagonal=1)
            W = self.P @ L @ (U + torch.diag(self.S))
            self.W_inv = torch.inverse(W)

        # x = z @ self.W_inv
        # x = (z.transpose(1, 2) @ self.W_inv).transpose(1, 2)
        x = torch.einsum('ij,bin->bjn', self.W_inv, z)

        # log_det = -torch.sum(torch.log(torch.abs(self.S)))
        # return x, log_det
        return x


# -----------------------------------------------------------------------------------------
class T_InvertibleConv1x1(nn.Module):

    def __init__(self, channel: int):
        super(T_InvertibleConv1x1, self).__init__()

        triangular_indices = InvertibleConv1x1.upper_trianglar_indices(channel - 1)
        triangular_indices = np.pad(triangular_indices, ((0, 1), (1, 0)))
        self.triangular_indices = torch.from_numpy(triangular_indices).long()  # [C, C]

        flat_dim = InvertibleConv1x1.n_elts_upper_triangular(channel)

        self.L_flat = nn.Parameter(torch.FloatTensor((flat_dim)))  # [flat_dim,]
        self.diag   = nn.Parameter(torch.FloatTensor((channel)))   # [C,]
        self.U_flat = nn.Parameter(torch.FloatTensor((flat_dim)))  # [flat_dim,]

        init.normal_(self.L_flat.data, mean=0.0, std=0.01)
        init.ones_(self.diag.data)
        init.normal_(self.U_flat.data, mean=0.0, std=0.01)

    def forward(self, x: Tensor):
        """
        x: [B, C, N]
        """
        L, d, U = self.get_LDU()

        z = torch.einsum('ij,bjn->bin', U, x)
        z = z * d.view(1, -1, 1)
        z = torch.einsum('ij,bjn->bin', L, z)

        logdet = torch.sum(d.abs().log(), dim=-1) * x.shape[-1]
        return z, logdet

    def inverse(self, z: Tensor):
        L, d, U = self.get_LDU()

        x, _ = torch.triangular_solve(z, L, upper=False, unitriangular=True)
        x = x / d.view(1, -1, 1)
        x, _ = torch.triangular_solve(x, U, upper=True,  unitriangular=True)

        return x

    def get_LDU(self):
        d_dim = self.diag.shape[-1]

        L = F.pad(self.L_flat, (1, 0))[self.triangular_indices]
        L = (L + torch.eye(d_dim, device=L.device)).T

        U = F.pad(self.U_flat, (1, 0))[self.triangular_indices]
        U = U + torch.eye(d_dim, device=U.device)

        return L, self.diag, U

    @staticmethod
    def upper_trianglar_indices(N: int):
        """
        # if N == 4, then it will generate 2d array like:
        [[10,  9,  8,  7],
         [ 0,  6,  5,  4],
         [ 0,  0,  3,  2],
         [ 0,  0,  0,  1]]
        """
        values = np.arange(N)

        idx = np.ogrid[:N, N:0:-1]
        idx = sum(idx) - 1

        mask = np.arange(N) >= np.arange(N)[:, None]
        return (idx + np.cumsum(values + 1)[:, None][::-1] - N + 1) * mask

    @staticmethod
    def n_elts_upper_triangular(N):
        return N * (N + 1) // 2 - 1


# # -----------------------------------------------------------------------------------------
# class InvertibleConv1x1A(nn.Module):
#     """Code modified from https://github.com/chaiyujin/glow-pytorch"""
# 
#     def __init__(self, num_channels, LU_decomposed=False):
#         super(InvertibleConv1x1A, self).__init__()
# 
#         w_shape = [num_channels, num_channels]
#         w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
#         if not LU_decomposed:
#             # Sample a random orthogonal matrix:
#             self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
#         else:
#             np_p, np_l, np_u = scipy.linalg.lu(w_init)
#             np_s = np.diag(np_u)
#             np_sign_s = np.sign(np_s)
#             np_log_s = np.log(np.abs(np_s))
#             np_u = np.triu(np_u, k=1)
#             l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
#             eye = np.eye(*w_shape, dtype=np.float32)
# 
#             self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
#             self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
#             self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
#             self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
#             self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
#             self.l_mask = torch.Tensor(l_mask)
#             self.eye = torch.Tensor(eye)
#         self.w_shape = w_shape
#         self.LU = LU_decomposed
# 
#     def get_weight(self, input, reverse):
#         B, _, _ = input.shape
#         w_shape = self.w_shape
#         if not self.LU:
#             pixels = input.numel() / B
#             dlogdet = torch.slogdet(self.weight)[1] * pixels
#             if not reverse:
#                 weight = self.weight.view(w_shape[0], w_shape[1], 1)
#             else:
#                 weight = torch.inverse(self.weight.double()).float()\
#                               .view(w_shape[0], w_shape[1], 1)
#             return weight, dlogdet
#         else:
#             self.p = self.p.to(input.device)
#             self.sign_s = self.sign_s.to(input.device)
#             self.l_mask = self.l_mask.to(input.device)
#             self.eye = self.eye.to(input.device)
#             l = self.l * self.l_mask + self.eye
#             u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
#             # dlogdet = thops.sum(self.log_s) * thops.pixels(input)
#             dlogdet = torch.sum(self.log_s) * (input.shape[1] * input.shape[2]) * input.new_ones((B,))
#             if not reverse:
#                 w = torch.matmul(self.p, torch.matmul(l, u))
#             else:
#                 l = torch.inverse(l.double()).float()
#                 u = torch.inverse(u.double()).float()
#                 w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
#             return w.view(w_shape[0], w_shape[1], 1), dlogdet
# 
#     def forward(self, x: Tensor):
#         """
#         log_det_J = log|abs(|W|)| * pixels
#         """
#         weight, dlogdet = self.get_weight(x, reverse=False)
#         # x = x.transpose(1, 2)
#         z = F.conv1d(x, weight)
#         # z = z.transpose(1, 2)
#         return z, dlogdet
# 
#     def inverse(self, z: Tensor):
#         weight, _ = self.get_weight(z, reverse=True)
#         # x = x.transpose(1, 2)
#         x = F.conv1d(z, weight)
#         # z = z.transpose(1, 2)
#         return x


# # -----------------------------------------------------------------------------------------
# class LinearCache(object):
#     """Helper class to store the cache of a linear transform.
# 
#     The cache consists of: the weight matrix, its inverse and its log absolute determinant.
#     """
# 
#     def __init__(self):
#         self.weight    = None
#         self.inverse   = None
#         self.log_det_J = None
# 
#     def invalidate(self):
#         self.weight    = None
#         self.inverse   = None
#         self.log_det_J = None


# # -----------------------------------------------------------------------------------------
# class InvertibleConv1x1B(nn.Module):
#     """An invertible 1x1 convolution with a fixed permutation, as introduced in the Glow paper.
#     Code modified from https://github.com/bayesiains/nsf
# 
#     Reference:
#     > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
#     """
# 
#     def __init__(self, channel: int, is_caching=False, is_identity_init=True, eps=1e-3):
#         super(InvertibleConv1x1B, self).__init__()
# 
#         self.eps     = eps
#         self.channel = channel
#         self.bias    = nn.Parameter(torch.FloatTensor(channel))
# 
#         self.is_caching = is_caching
#         if is_caching:
#             self.cache = LinearCache()
# 
#         self.tril_idx = np.tril_indices(channel, k=-1)
#         self.triu_idx = np.triu_indices(channel, k=1)
#         self.diag_idx = np.diag_indices(channel)
# 
#         n_triangular_entries = ((channel - 1) * channel) // 2
# 
#         self.tril_entries = nn.Parameter(torch.FloatTensor(n_triangular_entries))
#         self.triu_entries = nn.Parameter(torch.FloatTensor(n_triangular_entries))
#         self.unconstrained_upper_diag = nn.Parameter(torch.FloatTensor(channel))
# 
#         self._initialize(is_identity_init)
#         self.permutate = Permutation('random', channel)
# 
#     def forward(self, x: Tensor):
#         B, _, _ = x.shape
#         x = self.permutate(x)
#         x = x.transpose(1, 2)  # [B, N, C]
# 
#         if not self.training and self.is_caching:
#             self._check_forward_cache()
#             x = F.linear(x, self.cache.weight, self.bias)
#             log_det_J = self.cache.log_det_J * x.new_ones((B,))
#         else:
#             """Cost:
#                 output = O(D^2N)
#                 logabsdet = O(D)
#             where:
#                 D = num of features
#                 N = num of inputs
#             """
#             tril, triu = self._create_lower_upper()
#             x = F.linear(x, triu)
#             x = F.linear(x, tril, self.bias)
#             log_det_J = self._log_det_J() * x.new_ones((B,))
# 
#         return x.transpose(1, 2), log_det_J  # [B, C, N]
# 
#     def inverse(self, z: Tensor):
#         # B, C, N = z.shape
#         x = z.transpose(1, 2)  # [B, N, C]
# 
#         if not self.training and self.is_caching:
#             self._check_inverse_cache()
#             x = F.linear(x - self.bias, self.cache.inverse)
#             # log_det_J = (-self.cache.log_det_J) * x.new_ones((B,))
#         else:
#             """Cost:
#                 output = O(D^2N)
#                 logabsdet = O(D)
#             where:
#                 D = num of features
#                 N = num of inputs
#             """
#             tril, triu = self._create_lower_upper()
#             x = x - self.bias
#             x, _ = torch.triangular_solve(x.transpose(1, 2), tril, upper=False, unitriangular=True)
#             x, _ = torch.triangular_solve(x, triu, upper=True, unitriangular=False)
#             x = x.transpose(1, 2)
# 
#             # log_det_J = (-self._log_det_J()) * x.new_ones((B,))
# 
#         x = self.permutate.inverse(x.transpose(1, 2))
#         # return x, log_det_J
#         return x
# 
#     def _initialize(self, is_identity_init: bool):
#         init.zeros_(self.bias)
# 
#         if is_identity_init:
#             init.zeros_(self.tril_entries)
#             init.zeros_(self.triu_entries)
#             constant = np.log(np.exp(1 - self.eps) - 1)
#             init.constant_(self.unconstrained_upper_diag, constant)
#         else:
#             stdv = 1.0 / np.sqrt(self.channel)
#             init.uniform_(self.tril_entries, -stdv, stdv)
#             init.uniform_(self.triu_entries, -stdv, stdv)
#             init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)
#     
#     def _check_forward_cache(self):
#         if self.cache.weight is None:
#             self.cache.weight = self._weight()
#         if self.cache.log_det_J is None:
#             self.cache.log_det_J = self._log_det_J()
# 
#     def _check_inverse_cache(self):
#         if self.cache.inverse is None:
#             self.cache.inverse = self._weight_inverse()
#         if self.cache.log_det_J is None:
#             self.cache.log_det_J = self._log_det_J()
#     
#     def _weight(self):
#         """Cost:
#             weight = O(D^3)
#         where:
#             D = num of features
#         """
#         lower, upper = self._create_lower_upper()
#         return lower @ upper
#     
#     def _weight_inverse(self):
#         """Cost:
#             inverse = O(D^3)
#         where:
#             D = num of features
#         """
#         tril, triu = self._create_lower_upper()
#         identity = torch.eye(self.channel, self.channel, device=tril.device)
#         tril_inverse, _   = torch.triangular_solve(identity, tril, upper=False, unitriangular=True)
#         weight_inverse, _ = torch.triangular_solve(tril_inverse, triu, upper=True, unitriangular=False)
#         return weight_inverse
# 
#     def _log_det_J(self):
#         """Cost:
#             logabsdet = O(D)
#         where:
#             D = num of features
#         """
#         return torch.sum(torch.log(self.upper_diag))
# 
#     def _create_lower_upper(self):
#         tril = self.tril_entries.new_zeros(self.channel, self.channel)
#         tril[self.tril_idx[0], self.tril_idx[1]] = self.tril_entries
#         # The diagonal of L is taken to be all-ones without loss of generality.
#         tril[self.diag_idx[0], self.diag_idx[1]] = 1.0
# 
#         triu = self.triu_entries.new_zeros(self.channel, self.channel)
#         triu[self.triu_idx[0], self.triu_idx[1]] = self.triu_entries
#         triu[self.diag_idx[0], self.diag_idx[1]] = self.upper_diag
# 
#         return tril, triu
# 
#     @property
#     def upper_diag(self):
#         return F.softplus(self.unconstrained_upper_diag) + self.eps


# -----------------------------------------------------------------------------------------
# class InvertibleConv1x1(nn.Module):
# 
#     def __init__(self, num_channels, LU_decomposed=False):
#         super(InvertibleConv1x1, self).__init__()
# 
#         w_shape = [num_channels, num_channels]
#         w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
#         if not LU_decomposed:
#             # Sample a random orthogonal matrix:
#             self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
#         else:
#             np_p, np_l, np_u = scipy.linalg.lu(w_init)
#             np_s = np.diag(np_u)
#             np_sign_s = np.sign(np_s)
#             np_log_s = np.log(np.abs(np_s))
#             np_u = np.triu(np_u, k=1)
#             l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
#             eye = np.eye(*w_shape, dtype=np.float32)
# 
#             self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
#             self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
#             self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
#             self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
#             self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
#             self.l_mask = torch.Tensor(l_mask)
#             self.eye = torch.Tensor(eye)
#         self.w_shape = w_shape
#         self.LU = LU_decomposed
# 
#     def get_weight(self, input, reverse):
#         B, _, _ = input.shape
#         w_shape = self.w_shape
#         if not self.LU:
#             pixels = input.numel() / B
#             dlogdet = torch.slogdet(self.weight)[1] * pixels
#             if not reverse:
#                 weight = self.weight.view(w_shape[0], w_shape[1], 1)
#             else:
#                 weight = torch.inverse(self.weight.double()).float()\
#                               .view(w_shape[0], w_shape[1], 1)
#             return weight, dlogdet
#         else:
#             self.p = self.p.to(input.device)
#             self.sign_s = self.sign_s.to(input.device)
#             self.l_mask = self.l_mask.to(input.device)
#             self.eye = self.eye.to(input.device)
#             l = self.l * self.l_mask + self.eye
#             u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
#             dlogdet = thops.sum(self.log_s) * thops.pixels(input)
#             if not reverse:
#                 w = torch.matmul(self.p, torch.matmul(l, u))
#             else:
#                 l = torch.inverse(l.double()).float()
#                 u = torch.inverse(u.double()).float()
#                 w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
#             return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet
# 
#     def forward(self, x: Tensor):
#         """
#         log_det_J = log|abs(|W|)| * pixels
#         """
#         weight, dlogdet = self.get_weight(x, reverse=False)
#         x = x.transpose(1, 2)
#         z = F.conv1d(x, weight)
#         z = z.transpose(1, 2)
#         return z, dlogdet
# 
#     def inverse(self, x: Tensor):
#         weight, _ = self.get_weight(x, reverse=True)
#         x = x.transpose(1, 2)
#         z = F.conv1d(x, weight)
#         z = z.transpose(1, 2)
#         return z


# # -----------------------------------------------------------------------------------------
# class InvConv2dLU(nn.Module):
#     def __init__(self, in_channel):
#         super(InvConv2dLU, self).__init__()
# 
#         from scipy import linalg as la
# 
#         weight = np.random.randn(in_channel, in_channel)
#         q, _ = la.qr(weight)
#         w_p, w_l, w_u = la.lu(q.astype(np.float32))
#         w_s = np.diag(w_u)
#         w_u = np.triu(w_u, 1)
#         u_mask = np.triu(np.ones_like(w_u), 1)
#         l_mask = u_mask.T
# 
#         w_p = torch.from_numpy(w_p)
#         w_l = torch.from_numpy(w_l)
#         w_s = torch.from_numpy(w_s)
#         w_u = torch.from_numpy(w_u)
# 
#         logabs = lambda x: torch.log(torch.abs(x))
# 
#         self.register_buffer("w_p", w_p)
#         self.register_buffer("u_mask", torch.from_numpy(u_mask))
#         self.register_buffer("l_mask", torch.from_numpy(l_mask))
#         self.register_buffer("s_sign", torch.sign(w_s))
#         self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
#         self.w_l = nn.Parameter(w_l)
#         self.w_s = nn.Parameter(logabs(w_s))
#         self.w_u = nn.Parameter(w_u)
# 
#     def forward(self, input):
#         _, _, height, width = input.shape
# 
#         weight = self.calc_weight()
# 
#         out = F.conv2d(input, weight)
#         logdet = height * width * torch.sum(self.w_s)
# 
#         return out, logdet
# 
#     def calc_weight(self):
#         weight = (
#             self.w_p
#             @ (self.w_l * self.l_mask + self.l_eye)
#             @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
#         )
# 
#         return weight.unsqueeze(2).unsqueeze(3)
# 
#     def reverse(self, output):
#         weight = self.calc_weight()
# 
#         return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

# InvertibleConv1x1 = InvertibleConv1x1A
