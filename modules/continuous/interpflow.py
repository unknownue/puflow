
import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple, List

from pytorch3d.ops import knn_gather, knn_points

from modules.utils.probs import GaussianDistribution

from modules.continuous.odefunc import ODEfunc, ODEnet
from modules.continuous.normalization import MovingBatchNorm1d, IdentityBatchNorm
from modules.continuous.cnf import CNF
from modules.discrete.interpflow import InterpolationModule, FeatureExtractUnit, FeatMergeUnit


# -----------------------------------------------------------------------------------------
class FlowBlock(nn.Module):

    def __init__(self, idim, cdim, hdims: Tuple[int]=(512, 512, 512), batch_norm=True, layer_type='concatsquash', nonlinearity='tanh'):
        super(FlowBlock, self).__init__()

        self.bn_layer1 = MovingBatchNorm1d(idim) if batch_norm else IdentityBatchNorm()
        self.bn_layer2 = MovingBatchNorm1d(idim) if batch_norm else IdentityBatchNorm()

        diffeq = ODEnet(hdims, input_shape=(idim,), context_dim=cdim, layer_type=layer_type, nonlinearity=nonlinearity)
        odefunc = ODEfunc(diffeq)
        self.cnf = CNF(odefunc, True, T=0.5, train_T=True, solver='dopri5', atol=1e-5, rtol=1e-5)

    def forward(self, x: Tensor, c: Tensor):
        (B, N, _), device = x.shape, x.device
        logpx = torch.zeros((B, N, 1), device=device)

        x, logpx = self.bn_layer1(x, logpx=logpx, reverse=False)
        x, logpx = self.cnf(x, c, logpx, None, reverse=False, upratio=None)
        x, logpx = self.bn_layer2(x, logpx=logpx, reverse=False)

        logpx = torch.sum(logpx, dim=[1, 2], keepdim=False)
        return x, logpx

    def inverse(self, z: Tensor, c: Tensor, upratio: int):
        (B, N, _), device = z.shape, z.device
        logpx = torch.zeros((B, N, 1), device=device)

        z, logpx = self.bn_layer1(z, logpx=logpx, reverse=True)
        z, logpx = self.cnf(z, c, logpx, None, reverse=True, upratio=upratio)
        z, logpx = self.bn_layer2(z, logpx=logpx, reverse=True)

        return z

# -----------------------------------------------------------------------------------------
class PointInterpFlow(nn.Module):

    def __init__(self, pc_channel: int):
        super(PointInterpFlow, self).__init__()

        self.num_blocks = 6
        self.num_neighbors = 16

        self.dist   = GaussianDistribution(pc_channel, mu=0.0, vars=1.0, temperature=1.0)
        self.interp = InterpolationModule(pc_channel=3, k=self.num_neighbors)

        feat_channels = [pc_channel, 32, 64] + [128] * (self.num_blocks - 2)
        growth_widths = [8, 16] + [32] * (self.num_blocks - 2)
        cond_channels = [32, 64] + [128] * (self.num_blocks - 2)

        self.feat_convs = nn.ModuleList()
        for i in range(self.num_blocks):
            feat_conv = FeatureExtractUnit(feat_channels[i], feat_channels[i + 1], self.num_neighbors, growth_widths[i], is_dynamic=False)
            self.feat_convs.append(feat_conv)

        self.merge_convs = nn.ModuleList()
        for i in range(self.num_blocks):
            merge_unit = FeatMergeUnit(feat_channels[i + 1], cond_channels[i])
            self.merge_convs.append(merge_unit)

        self.flow_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            step = FlowBlock(idim=pc_channel, cdim=cond_channels[i], hdims=tuple([64] * 2), batch_norm=False)  # TODO: Try batch norm
            self.flow_blocks.append(step)

    def feat_extract(self, xyz: Tensor, knn_idx: Tensor):
        cs = []
        c = xyz

        for i in range(self.num_blocks):
            c = self.feat_convs[i](c, knn_idx=knn_idx)
            _c = self.merge_convs[i](c)
            cs.append(_c)
            # cs.append(None)
        return cs

    def f(self, xyz: Tensor, cs: List[Tensor]):
        (B, _, _), device = xyz.shape, xyz.device
        log_det_J = torch.zeros((B,), device=device)

        p = xyz

        for i in range(self.num_blocks):
            p, _log_det_J = self.flow_blocks[i](p, cs[i])
            if _log_det_J is not None:
                log_det_J += _log_det_J

        return p, log_det_J

    def g(self, z: Tensor, cs: Tensor, upratio: int):
        z = torch.flatten(z.transpose(2, 3), 1, 2)

        for i in reversed(range(self.num_blocks)):
            c = torch.repeat_interleave(cs[i], upratio, dim=1)
            z = self.flow_blocks[i].inverse(z, c, upratio)
        return z

    def set_to_initialized_state(self):
        pass

    def forward(self, xyz: Tensor, upratio=4):
        _, knn_idx, _ = knn_points(xyz, xyz, K=self.num_neighbors, return_nn=False, return_sorted=False)  # [B, N, K]

        p = xyz

        cs = self.feat_extract(p, knn_idx)
        z, logp_x = self.log_prob(p, cs)

        fz = self.interp(z, xyz, upratio)
        x = self.g(fz, cs, upratio)
        return x, logp_x

    def log_prob(self, xyz: Tensor, cs: List[Tensor]):
        x, log_det_J = self.f(xyz, cs)

        logp_x = self.dist.standard_logp(x).to(x.device)
        logp_x = -torch.mean(logp_x - log_det_J)
        return x, logp_x

    def sample(self, sparse: Tensor, upratio=4):
        dense, _ = self(sparse, upratio)
        return dense
# -----------------------------------------------------------------------------------------
