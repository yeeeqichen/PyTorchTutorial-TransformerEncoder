#!/usr/bin/python3
# encoding: utf-8
"""
@author: yeeeqichen
@contact: 1700012775@pku.edu.cn
@file: DataLoader.py
@time: 2020/8/11 3:10 下午
@desc:
"""
from Config import config
import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
config.vocab_size = len(TEXT.vocab.stoi)


def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(-1, bsz)
    return data.to(config.device)


def get_batch(data, i):
    """
    返回一个训练实例，其中target是src每个位置的词的下一个词
    src的shape为 (sequence, batch) target的shape为 (sequence * batch)，这是为了后续计算交叉熵
    :param data: 数据集 (-1, batch)
    :param i: 目前batch的起始位置
    :return: 输入src，目标target
    """
    seq_len = min(config.seq_len, len(data) - i - 1)
    src = data[i:i + seq_len]
    target = data[i + 1:i + 1 + seq_len].view(-1)
    return src, target


train_data = batchify(train_txt, config.batch_size)
val_data = batchify(val_txt, config.batch_size)
test_data = batchify(test_txt, config.batch_size)
print(train_data.shape)
