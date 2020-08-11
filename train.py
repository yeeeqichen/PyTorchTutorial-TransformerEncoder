#!/usr/bin/python3
# encoding: utf-8
"""
@author: yeeeqichen
@contact: 1700012775@pku.edu.cn
@file: train.py
@time: 2020/8/11 3:41 下午
@desc:
"""
from Config import config
from DataLoader import train_data, val_data, test_data, TEXT, get_batch
from Model import model
import torch
import math
import time


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=config.lr, params=model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, config.batch_size)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.view(-1, config.vocab_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                    epoch, batch, len(train_data) // config.batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, config.batch_size):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, config.vocab_size)
            total_loss += len(data) * loss_fn(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


for epoch in range(config.EPOCH):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < config.best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
