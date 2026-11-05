"""Frozen recurrent scratch policy plus learned withdrawal ACT decoder.

The binary router reads causal learned memory, never a ground-truth phase.
Two action chunk decoder branches share the frozen continuous encoder/memory.
The control uses exactly the same learned semantic head but the original decoder.
"""
import copy
import torch
from torch import nn
from act_recurrent_joint_phase_v1 import ACT as OriginalACT
class ACT(nn.Module):
    def __init__(self,chunk=24,input_dim=64,width=128):
        super().__init__()
        self.base=OriginalACT(chunk,input_dim,width)
        self.withdrawal=OriginalACT(chunk,input_dim,width)
        self.phase=copy.deepcopy(self.base.phase)
        self.router=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,2))
        nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)
        self.use_specialist=True
        for module in (self.base,self.withdrawal):
            for p in module.parameters():p.requires_grad_(False)
        for name,p in self.withdrawal.named_parameters():
            if name in ('query','prior') or name.startswith(('decoder.','out.','memory_projection.')):p.requires_grad_(True)
    def initialize(self,sd):
        self.base.load_state_dict(sd);self.withdrawal.load_state_dict(sd)
        self.phase.load_state_dict(self.base.phase.state_dict())
    def encode(self,x,lengths):
        from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
        with torch.no_grad():
            embedded=self.base.state(x)
            h,_=self.base.memory(pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False))
            hidden,_=pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        return embedded,hidden
    def sequence(self,x,lengths,bi,ti):
        embedded,h=self.encode(x,lengths)
        return self.decode(embedded[bi,ti],h[bi,ti]),self.phase(h),h
    def decode(self,current,h):
        with torch.no_grad():original=self.base.decode(current,h)
        alternative=self.withdrawal.decode(current,h)
        if not self.use_specialist:return original
        choose=self.router(h).argmax(-1).bool()
        return torch.where(choose[:,None,None],alternative,original)
    def observe(self,x,hidden=None):
        with torch.no_grad():hidden,_=self.base.observe(x,hidden)
        return hidden,self.phase(hidden[-1])
    def predict(self,x,hidden):return self.decode(self.base.state(x),hidden[-1])
    def route_probabilities(self,hidden):return self.router(hidden[-1]).softmax(-1)
