#!/usr/bin/python3
# encoding: utf-8
"""
@author: yeeeqichen
@contact: 1700012775@pku.edu.cn
@file: Config.py
@time: 2020/8/11 2:22 下午
@desc:
"""
import torch


class Config:
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 10
        self.num_layers = 3
        self.embed_size = 128
        self.vocab_size = 1
        self.num_heads = 4
        self.hidden_size = 256
        self.max_len = 20
        self.seq_len = 20
        self.dropout = 0.2
        self.EPOCH = 3
        self.best_val_loss = float('inf')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = Config()
