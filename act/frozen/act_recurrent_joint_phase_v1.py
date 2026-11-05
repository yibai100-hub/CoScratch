"""Causal recurrent ACT-style action chunks with model-predicted phase.

64 continuous observation features; no phase one-hot, wall clock, source frame,
future action, manual phase counter, or source trajectory at deployment.
Memory is observed every12 actual120Hz feedback samples, independent of the
action replan prefix12/24. Phase logits do not clamp or gate action predictions.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
class ACT(nn.Module):
    def __init__(self,chunk=24,input_dim=64,width=128):
        super().__init__();self.chunk=chunk;self.input_dim=input_dim
        self.state=nn.Linear(input_dim,width)
        self.memory=nn.GRU(width,width,batch_first=True)
        self.memory_projection=nn.Linear(width,width)
        nn.init.zeros_(self.memory_projection.weight);nn.init.zeros_(self.memory_projection.bias)
        self.prior=nn.Parameter(torch.zeros(width))
        self.query=nn.Parameter(torch.randn(1,chunk,width)*.01)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,4,width*4,dropout=0,batch_first=True),2)
        self.out=nn.Linear(width,17)
        self.phase=nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Linear(128,14))
    def decode(self,current,hidden):
        tokens=torch.stack([current,self.prior[None]+self.memory_projection(hidden)],dim=1)
        return self.out(self.decoder(self.query.expand(len(current),-1,-1),tokens))
    def sequence(self,x,lengths,batch_indices,tick_indices):
        embedded=self.state(x)
        packed=pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        memory,_=self.memory(packed)
        hidden,_=pad_packed_sequence(memory,batch_first=True,total_length=x.shape[1])
        prediction=self.decode(embedded[batch_indices,tick_indices],hidden[batch_indices,tick_indices])
        return prediction,self.phase(hidden),hidden
    def observe(self,x,hidden=None):
        current=self.state(x);sequence,hidden=self.memory(current[:,None],hidden)
        return hidden,self.phase(sequence[:,0])
    def predict(self,x,hidden):return self.decode(self.state(x),hidden[-1])
