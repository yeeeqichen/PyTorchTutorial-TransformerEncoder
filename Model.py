#!/usr/bin/python3
# encoding: utf-8
"""
@author: yeeeqichen
@contact: 1700012775@pku.edu.cn
@file: Model.py
@time: 2020/8/11 2:22 下午
@desc:
"""
import torch
import math
from Config import config


class Model(torch.nn.Module):
    """
    TransformerEncoder 语言模型
    输入: (sequence, batch),元素的值域为vocab_size，相当于一列是一个句子
    输出: (sequence, batch, vocab_size)，预测每个位置的下一个词
    """
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.TransformerEncoderLayer(config.embed_size, config.num_heads,
                                                      config.hidden_size, config.dropout)
        self.encoder = torch.nn.TransformerEncoder(self.layer, config.num_layers)
        self.decoder = torch.nn.Linear(config.embed_size, config.vocab_size)
        self.embed = torch.nn.Embedding(config.vocab_size, config.embed_size)
        self.positional_embed = PositionalEncode()
        self.src_mask = None
        self.init_weight()

    def init_weight(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    @staticmethod
    def _generate_square_subsequent_mask(size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        # print(mask)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(config.device)

    def forward(self, inputs):
        if self.src_mask is None or self.src_mask.size(0) != len(inputs):
            self.src_mask = self._generate_square_subsequent_mask(len(inputs))
        src = self.embed(inputs) * math.sqrt(config.embed_size)
        src = self.positional_embed(src)
        encode = self.encoder(src, self.src_mask)
        output = self.decoder(encode)
        return output


class PositionalEncode(torch.nn.Module):
    """
    PositionalEncode 为embedding加入位置信息
    输入: (sequence, batch, embed_size)
    输出: (sequence, batch, embed_size)
    """
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(config.dropout)
        pe = torch.zeros(config.max_len, config.embed_size)
        position = torch.arange(0, config.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embed_size, 2).float() * (-math.log(10000.0) / config.embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


model = Model().to(config.device)
print(model)


