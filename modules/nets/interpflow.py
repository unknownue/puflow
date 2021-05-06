
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F

from modules.nets.dgcnn import DynamicGraphCNN
# from modules.nets.layer import SoftClampling, ResidualNet1D

from modules.flows.coupling import AffineCouplingLayer, AffineInjectorLayer
from modules.flows.coupling import AffineSpatialCouplingLayer
from modules.flows.normalize import ActNorm
from modules.flows.permutate import Permutation

from modules.utils.probs import GaussianDistribution


# -----------------------------------------------------------------------------------------
class DistanceEncoder(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, k: int):
        super(DistanceEncoder, self).__init__()

        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(dim_in * 3 + 1, 64, [1, 1]),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, [1, 1]),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, dim_out, [1, 1]))

    def distance_vec(self, xyz: Tensor):
        B, N, C = xyz.shape
        idxb = torch.arange(B).view(-1, 1)

        Txyz = xyz.transpose(1, 2)   # [B, C, N]
        idx_knn = DynamicGraphCNN.knn(Txyz, k=self.k)  # [B, N, k]
        idx_knn = idx_knn.view(B, -1)                  # [B, N * k]
        neighbors = xyz[idxb, idx_knn].view(B, N, self.k, C)  # [B, N, k, C]

        pt_raw = torch.unsqueeze(xyz, dim=2).repeat(1, 1, self.k, 1)  # [B, N, k, C]
        neighbor_vector = pt_raw - neighbors   # [B, N, k, C]
        distance = torch.sqrt(torch.sum(neighbor_vector ** 2, dim=-1, keepdim=True))

        f_distance = torch.cat([pt_raw, neighbors, neighbor_vector, distance], dim=-1)
        f_distance = f_distance.permute(0, 3, 1, 2)
        return f_distance, idx_knn  # [B, C, N, k]

    def forward(self, xyz: Tensor):
        f_dist, idx_knn = self.distance_vec(xyz)  # [B, C, N, k]
        f = self.mlp(f_dist)
        return f, idx_knn  # [B, C_out, N, k]


# -----------------------------------------------------------------------------------------
class EdgeConv(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, k: int, dim_h=64, n_conv=1, is_knn=True):
        super(EdgeConv, self).__init__()
        self.k      = k
        self.n_conv = n_conv
        self.is_knn = is_knn

        self.convs = nn.ModuleList()
        for i in range(n_conv):
            indim = dim_in * 2 if i == 0 else dim_h
            conv = nn.Sequential(
                nn.Conv2d(indim, dim_h, [1, 1], bias=True),
                nn.BatchNorm2d(dim_h),
                nn.ReLU(inplace=True))
            self.convs.append(conv)
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim_h, dim_out, [1, 1], bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True))

    def forward(self, x: Tensor):
        if self.is_knn:
            x = DynamicGraphCNN.get_graph_feature(x, self.k)
        for i in range(self.n_conv):
            x = self.convs[i](x)
        x = self.conv_out(x)
        return x


# -----------------------------------------------------------------------------------------
class LinearA1D(nn.Module):

    def __init__(self, dim_in: int, dim_h: int, dim_out: int, dim_c=None):
        super(LinearA1D, self).__init__()

        linear_zero = nn.Conv1d(dim_h, dim_out, kernel_size=1)
        linear_zero.weight.data.zero_()
        linear_zero.bias.data.zero_()

        in_channel = dim_in if dim_c is None else dim_in + dim_c

        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, dim_h, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim_h),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(dim_h, dim_h, kernel_size=1, bias=True),
            nn.LeakyReLU(inplace=True),
            linear_zero)

    def forward(self, h: Tensor, c: Tensor=None):
        if c is not None:
            h = torch.cat([h, c], dim=1)
        h = self.layers(h)
        return h


# -----------------------------------------------------------------------------------------
class FlowStep(nn.Module):

    def __init__(self, idim, hdim, cdim, is_even):
        super(FlowStep, self).__init__()

        self.actnorm   = ActNorm(idim)
        self.permutate = Permutation('inv1x1', idim)   # inv1x1 > random

        if idim == 3:
            tdim = 1 if is_even else 2
            self.coupling1 = AffineSpatialCouplingLayer('affine', LinearA1D, split_dim=1, is_even=is_even, clamp=None,
                params={ 'dim_in' : tdim, 'dim_h': hdim, 'dim_out': idim - tdim, 'dim_c': cdim })
            # self.coupling1 = AffineSpatialCouplingLayer('affine', ResidualNet1D, split_dim=1, clamp=SoftClampling(clamp=1.9), is_even=is_even,
            #     params={ 'in_channel': tdim, 'hidden_channel': hdim, 'out_channel': idim - tdim, 'num_blocks': 2, 'u_channel': cdim })
        else:
            self.coupling1 = AffineCouplingLayer('affine', LinearA1D, split_dim=1, clamp=None,
                params={ 'dim_in' : idim // 2, 'dim_h': hdim, 'dim_out': idim - idim // 2, 'dim_c': cdim })
            # self.coupling1 = AffineCouplingLayer('affine', ResidualNet1D, split_dim=1, clamp=SoftClampling(clamp=1.9),
            #     params={ 'in_channel': idim // 2, 'hidden_channel': hdim, 'out_channel': idim - idim // 2, 'num_blocks': 2, 'u_channel': cdim })

        self.coupling2 = AffineInjectorLayer('affine', LinearA1D, split_dim=1, clamp=None,
            params={ 'dim_in': cdim, 'dim_h': hdim, 'dim_out': idim, 'dim_c': None })
        # self.coupling2 = AffineInjectorLayer('affine', ResidualNet1D, split_dim=1, clamp=SoftClampling(clamp=1.9),
        #     params={ 'in_channel': cdim, 'hidden_channel': hdim, 'out_channel': idim, 'num_blocks': 2, 'u_channel': None })

    def forward(self, x: Tensor, c: Tensor=None):
        x, _log_det0 = self.actnorm(x)
        x, _log_det1 = self.permutate(x)
        x, _log_det2 = self.coupling1(x, c)
        x, _log_det3 = self.coupling2(x, c)

        return x, _log_det0 + _log_det1 + _log_det2 + _log_det3

    def inverse(self, z: Tensor, c: Tensor=None):
        z = self.coupling2.inverse(z, c)
        z = self.coupling1.inverse(z, c)
        z = self.permutate.inverse(z)
        z = self.actnorm.inverse(z)
        return z


# -----------------------------------------------------------------------------------------
class KnnContextEncoder(nn.Module):
    def __init__(self, pc_channel, k):
        super(KnnContextEncoder, self).__init__()
        self.distance_encoder = DistanceEncoder(dim_in=pc_channel, dim_out=128, k=k)
        self.edge_convolution = EdgeConv(dim_in=pc_channel, dim_out=128, k=k, dim_h=64, n_conv=2)

    def forward(self, xyz: Tensor):
        dist, idx_knn = self.distance_encoder(xyz)  # [B, 128, N, k]
        edge = self.edge_convolution(xyz.transpose(1, 2))  # [B, 128, N, k]
        return torch.cat([dist, edge], dim=1), idx_knn

# -----------------------------------------------------------------------------------------
class InterpolationModule(nn.Module):

    def __init__(self, pc_channel, k: int):
        super(InterpolationModule, self).__init__()

        self.k = k
        self.r_max = 16
        self.mlp = nn.Sequential(
            nn.Conv2d(256, 128, [1, 1]),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 64, [1, 1]),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, self.r_max, [1, 1]))
        self.knn_context = KnnContextEncoder(pc_channel, k)


    def forward(self, z: Tensor, xyz: Tensor, upratio: int):

        (B, C, N), device = z.shape, z.device
        idxb = torch.arange(B, device=device).view(-1, 1, 1)
        # idxn = torch.arange(N, device=device).view(1, -1, 1)
        # randomly select solution from r_max 
        # idxr = torch.rand((B, N, self.r_max), device=device).argsort(-1)[:, :, :upratio]
        # assert upratio <= self.r_max

        # Learn interpolation weight for each point
        context, idx_knn = self.knn_context(xyz)   # [B, C, N, k], [B, N * k]
        weights = self.mlp(context)           # [B, r_max, N, k]
        weights = weights.permute(0, 2, 1, 3) # [B, N, r_max, k]
        # weights = F.softmax(weights[idxb, idxn, idxr], dim=-1)  # [B, N, upratio, k]
        weights = F.softmax(weights[:, :, :upratio], dim=-1)  # [B, N, upratio, k]

        # Interpolation
        nei_prior = z.transpose(1, 2)[idxb.view(-1, 1), idx_knn].view(B, N, self.k, C)  # [B, N, k, C]
        nei_prior = nei_prior.permute(0, 1, 3, 2)           # [B, N, C, k]
        intep_prior = torch.einsum('bnck,bnrk->bncr', nei_prior, weights)
        return intep_prior


class PointMerge(nn.Module):
    def __init__(self, dim_out: int, c1: int, c2: int, c3: int):
        super().__init__()
        self.max_conv = nn.Sequential(
            nn.Conv2d(c1 + c2 + c3, 256, [1, 1]),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(256 + c1, dim_out, [1, 1]))

    def forward(self, x: Tensor, m1: Tensor, m2: Tensor):
        _, _, N, M = x.shape
        _x = self.max_conv(torch.cat([x, m1, m2], dim=1))  # [B, C, N, M]
        mx = torch.max_pool2d(_x, [N, M])  # [B, C, 1, 1]
        xglobal = mx.repeat(1, 1, N, 1)   # [B, C, N, 1]
        x = self.out_conv(torch.cat([x, xglobal], dim=1))
        return x


class MyEdgeConv(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, k: int, n_conv=1, is_pooling=True, is_knn=True):
        super().__init__()
        self.k          = k
        self.n_conv     = n_conv
        self.is_knn     = is_knn
        self.is_pooling = is_pooling

        self.convs = nn.ModuleList()
        for i in range(n_conv):
            indim = dim_in * 2 if i == 0 else 64
            conv = nn.Sequential(
                nn.Conv2d(indim, 64, [1, 1], bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            self.convs.append(conv)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64 * 2, dim_out, [1, 1], bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True))

    def forward(self, x: Tensor):
        if self.is_knn:
            x = DynamicGraphCNN.get_graph_feature(x, self.k)
        for i in range(self.n_conv):
            x = self.convs[i](x)

        if self.is_pooling:
            m1, _ =  torch.max(x, dim=-1, keepdim=True)
            m2    = torch.mean(x, dim=-1, keepdim=True)
            x = self.conv_out(torch.cat([m1, m2], dim=1))
            return x, m1, m2
        else:
            return x


# -----------------------------------------------------------------------------------------
class PointInterpFlow(nn.Module):

    def __init__(self, pc_channel: int, num_neighbors: int):
        super(PointInterpFlow, self).__init__()

        self.nsteps = 10

        self.dist    = GaussianDistribution(pc_channel, mu=0.0, vars=1.0, temperature=1.0)
        self.interp  = InterpolationModule(pc_channel, k=num_neighbors)

        conv_dim = 256
        self.edge_convs  = nn.ModuleList([MyEdgeConv(pc_channel, 64, k=num_neighbors, n_conv=2)] + [MyEdgeConv(64, 64, k=num_neighbors, n_conv=1) for _ in range(self.nsteps - 1)])
        self.merge_convs = nn.ModuleList([PointMerge(conv_dim, 64, 64, 64) for _ in range(self.nsteps)])

        self.flow_steps = nn.ModuleList()
        for i in range(self.nsteps):
            step = FlowStep(idim=pc_channel, hdim=64, cdim=256, is_even=(i % 2 == 0))
            self.flow_steps.append(step)

    def f(self, xyz: Tensor, **kwargs):
        (B, N, _), device = xyz.shape, xyz.device
        log_det_J = torch.zeros((B,), device=device)

        p = xyz.transpose(1, 2)  # [B, C, N]
        c = p
        cs = []

        for i in range(self.nsteps):
            c, m1, m2 = self.edge_convs[i](c)
            _c = torch.squeeze(self.merge_convs[i](c, m1, m2), dim=-1)
            cs.append(_c)
            p, _log_det_J = self.flow_steps[i](p, _c)
            if _log_det_J is not None:
                log_det_J += _log_det_J

        return p, cs, log_det_J

    def g(self, z: Tensor, cs: Tensor, upratio: int):
        z = torch.flatten(z.permute(0, 2, 1, 3), 2)

        for i in reversed(range(self.nsteps)):
            _c = cs[i]
            _c = torch.repeat_interleave(_c, upratio, dim=-1)
            z = self.flow_steps[i].inverse(z, _c)

        z = torch.transpose(z, 1, 2)
        return z
    
    def set_to_initialized_state(self):
        for i in range(self.nsteps):
            self.flow_steps[i].actnorm.is_inited = True

    def forward(self, p: Tensor, upratio=4):
        z, c, logp_x = self.log_prob(p)
        fz = self.interp(z, p, upratio)
        x = self.g(fz, c, upratio)
        return x, logp_x

    def log_prob(self, p: Tensor, **kwargs):
        x, c, log_det_J = self.f(p, **kwargs)

        # standard gaussian probs
        logp_x = self.dist.standard_logp(x.transpose(1, 2)).to(x.device)
        logp_x = -torch.mean(logp_x + log_det_J)
        return x, c, logp_x

    def sample(self, sparse: Tensor, upratio=4, **kwargs):
        x, _ = self(sparse, upratio)
        return x
# -----------------------------------------------------------------------------------------
