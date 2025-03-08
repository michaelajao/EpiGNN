# -*- coding: utf-8 -*-

# Import necessary modules
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from utils import *
from layers import *
from ablation import WOGlobal
from ablation import WOLocal
from ablation import WORAGL
from ablation import baseline

class EpiGNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # Initialize the EpiGNN model with provided arguments and data.
        # Arguments setting
        self.adj = data.adj  # Adjacency matrix representing spatial relationships
        self.m = data.m  # Number of regions
        self.w = args.window  # Input sequence length (window size)
        self.n_layer = args.n_layer  # Number of GCN layers
        self.droprate = args.dropout  # Dropout rate for regularization
        self.hidR = args.hidR  # Hidden dimension for region-aware convolution
        self.hidA = args.hidA  # Hidden dimension for attention layers
        self.hidP = args.hidP  # Hidden dimension for adaptive pooling
        self.k = args.k  # Number of kernels in convolutional layers
        self.s = args.s  # Kernel size for temporal convolution
        self.n = args.n  # Number of GCN layers
        self.res = args.res  # Residual connection flag
        self.hw = args.hw  # Highway network flag
        self.dropout = nn.Dropout(self.droprate)  # Dropout layer
        
        # Highway network for autoregressive component
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)

        # Check for external features
        if args.extra:
            self.extra = True
            self.external = data.external
        else:
            self.extra = False

        # Feature embedding using Region-Aware Convolution
        self.hidR = self.k * 4 * self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)

        # Global transmission risk encoding
        self.WQ = nn.Linear(self.hidR, self.hidA)  # Query projection
        self.WK = nn.Linear(self.hidR, self.hidA)  # Key projection
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)  # Temporal encoding

        # Local transmission risk encoding
        self.degree = data.degree_adj  # Degree adjacency vector
        self.s_enc = nn.Linear(1, self.hidR)  # Spatial encoding

        # External resources (if any)
        self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m))

        # Graph Generator and GCN layers
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m))
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList([
            GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
            for _ in range(self.n)
        ])

        # Prediction layer
        if self.res == 0:
            self.output = nn.Linear(self.hidR * 2, 1)
        else:
            self.output = nn.Linear(self.hidR * (self.n + 1), 1)

        self.init_weights()  # Initialize model weights

    def init_weights(self):
        # Initialize weights of the model using Xavier uniform distribution
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, index, isEval=False):
        # Forward pass of the model
        # x: Input data tensor of shape (batch_size, sequence_length, num_regions)
        # index: Indices of the data samples
        # isEval: Boolean flag indicating evaluation mode

        batch_size = x.shape[0]  # Get batch size

        # Step 1: Feature embedding using Region-Aware Convolution
        temp_emb = self.backbone(x)  # Shape: (batch_size, num_regions, hidR)

        # Step 2: Generate global transmission risk encoding
        query = self.WQ(temp_emb)  # Project features to query space
        query = self.dropout(query)
        key = self.WK(temp_emb)    # Project features to key space
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))  # Compute attention scores
        #attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # Step 3: Generate local transmission risk encoding
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion
        feat_emb = temp_emb + t_enc + s_enc

        # Step 4: Region-Aware Graph Learner
        # Load external resource
        if self.extra:
            extra_adj_list=[]
            zeros_mt = torch.zeros((self.m, self.m)).to(self.adj.device)
            #print(self.external.shape)
            for i in range(batch_size):
                offset = 20
                if i-offset>=0:
                    idx = i-offset
                    extra_adj_list.append(self.external[index[i],:,:].unsqueeze(0))
                else:
                    extra_adj_list.append(zeros_mt.unsqueeze(0))
            extra_info = torch.concat(extra_adj_list, dim=0) # [1872, 52]
            extra_info = extra_info
            #print(extra_info.shape) # batch_size, self.m self.m
            external_info = torch.mul(self.external_parameter, extra_info)
            external_info = F.relu(external_info)
            #print(self.external_parameter)

        # Apply Graph Learner to generate a graph
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, self.adj)
        adj = self.graphGen(temp_emb)
        
        # If additional information => fusion
        if self.extra:
            adj = adj + spatial_adj + external_info
        else:
            adj = adj + spatial_adj

        # Get laplace adjacent matrix
        laplace_adj = getLaplaceMat(batch_size, self.m, adj)
        
        # Graph Convolution Network
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)
        '''
        if self.res == 1:
            node_state = torch.cat(node_state_list, dim=-1)
        
        node_state = self.GCNBlock1(feat_emb, laplace_adj)
        node_state = self.dropout(node_state)
        node_state = self.GCNBlock2(node_state, laplace_adj)
        node_state = self.dropout(node_state)
        '''

        # Final prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # Highway means autoregressive model
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        
        # If evaluation, return some intermediate results
        if isEval:
            imd = (adj, attn)
        else:
            imd = None

        # Return the final output and any intermediate results
        return res, imd
