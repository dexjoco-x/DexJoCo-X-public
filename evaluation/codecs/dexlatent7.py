"""Project full-native-state DexLatent32 heads; arm channels are untouched.

Architecture follows the existing project DexLatent adapter. Unlike its old
Inspire6 head, the new Inspire12 head can observe real passive-joint deviations.
"""
import numpy as np
import torch
from torch import nn


def mlp(n,hidden,out=None):
    layers=[]
    for width in hidden:
        layers.extend([nn.Linear(n,width),nn.LayerNorm(width),nn.ReLU()]);n=width
    if out is not None:layers.extend([nn.Linear(n,out),nn.Tanh()])
    return nn.Sequential(*layers)


class HandHead(nn.Module):
    def __init__(self,dof):
        super().__init__();self.hand_encoder_backbone=mlp(dof,[64,128,64])
        self.hand_mean_head=nn.Linear(64,32);self.hand_logvar_head=nn.Linear(64,32)
        self.hand_decoder=mlp(32,[64,128,64],dof)
        self.residual_decoder=nn.Sequential(nn.Linear(32,256),nn.LayerNorm(256),nn.ReLU(),
            nn.Linear(256,512),nn.LayerNorm(512),nn.ReLU(),nn.Linear(512,256),nn.LayerNorm(256),
            nn.ReLU(),nn.Linear(256,dof))
        nn.init.zeros_(self.residual_decoder[-1].weight);nn.init.zeros_(self.residual_decoder[-1].bias)

    def encode(self,q):return self.hand_mean_head(self.hand_encoder_backbone(q))
    def decode_raw(self,z):return self.hand_decoder(z)+self.residual_decoder(z)
    def decode(self,z):return self.decode_raw(z).clamp(-1,1)


class JointNormalizer(nn.Module):
    def __init__(self,contract):
        super().__init__();bounds=torch.tensor(contract['joint_limits'],dtype=torch.float32)
        self.register_buffer('lower',bounds[:,0]);self.register_buffer('width',bounds[:,1]-bounds[:,0])
    def encode(self,q):
        # Match the upstream DexLatent contract: simulator/contact dynamics can
        # push observed qpos slightly beyond the static XML joint limits.
        return (2*(q-self.lower)/self.width-1).clamp(-1,1)
    def decode(self,q):return (q+1)*self.width/2+self.lower


class DexLatent7:
    def __init__(self,contract,head):
        self.head=head.eval().requires_grad_(False);self.norm=JointNormalizer(contract)
        self.expand=np.array(contract['control_to_full'],dtype=np.float32)
        self.bias=np.array(contract['mimic_bias'],dtype=np.float32)
        self.control=np.array(contract['full_to_control'],dtype=np.float32)
    @torch.inference_mode()
    def state(self,q):
        return self.head.encode(self.norm.encode(torch.tensor(np.asarray(q),dtype=torch.float32))).numpy()
    def action(self,u):return self.state(np.asarray(u)@self.expand.T+self.bias)
    @torch.inference_mode()
    def decode(self,z):
        q=self.norm.decode(self.head.decode(torch.as_tensor(z,dtype=torch.float32))).numpy()
        return q@self.control.T


class SidedDexLatent7:
    """Canonicalize a mirrored side before/after the right-trained head."""
    def __init__(self, canonical_contract, side_contract, head):
        from faas7 import FAAS7
        self.canonical = DexLatent7(canonical_contract, head)
        self.canonical_faas = FAAS7(canonical_contract)
        self.side_faas = FAAS7(side_contract)

    def state(self, q):
        q = self.canonical_faas.decode_state(self.side_faas.state(q))
        return self.canonical.state(q)

    def action(self, u):
        u = self.canonical_faas.decode(self.side_faas.action(u))
        return self.canonical.action(u)

    def decode(self, z):
        u = self.canonical.decode(z)
        return self.side_faas.decode(self.canonical_faas.action(u))
