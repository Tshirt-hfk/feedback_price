import argparse
import datetime
import time

import torch
import yaml
import logging
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LongformerConfig

from dataloader import load_data, FPDataset, num_labels, id2label, label2id
from inverseSquareRootSchedule import InverseSquareRootSchedule
from labelSmoothedCrossEntropy import LabelSmoothedCrossEntropyCriterion
from model import LongformerSoftmaxForNer
from utils import Arguments


# 定义评价指标
def compute_metrics(predictions, labels, masks):
    predictions = sum([[p for (p, m) in zip(prediction, mask) if m] for prediction, mask in
                       zip(predictions, masks)], [])
    labels = sum([[l for (l, m) in zip(labels, mask) if m] for labels, mask in
                  zip(labels, masks)], [])
    accuracy = metrics.accuracy_score(y_pred=predictions, y_true=labels)
    precision = metrics.precision_score(y_pred=predictions, y_true=labels, average='macro')
    recall = metrics.recall_score(y_pred=predictions, y_true=labels, average='macro')
    f1 = metrics.f1_score(y_pred=predictions, y_true=labels, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    data = load_data(tokenizer)
    train_dataset = FPDataset(data, tokenizer, batch_size=args.batch_size, train=True)
    valid_dataset = FPDataset(data, tokenizer, batch_size=args.batch_size, train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=None, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=None)
    config = LongformerConfig.from_pretrained("allenai/longformer-base-4096", num_labels=num_labels)
    model = LongformerSoftmaxForNer.from_pretrained("allenai/longformer-base-4096", config=config).cuda()
    criterion = LabelSmoothedCrossEntropyCriterion(eps=args.label_smoothing)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, betas=(args.betas0, args.betas1), weight_decay=args.weight_decay)
    lr_schedule = InverseSquareRootSchedule(optimizer, lr=args.lr, warmup_updates=args.warmup_updates)

    zero_time = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train(train_loader, model, criterion, optimizer, lr_schedule)
        val_loss, val_metric = validate(valid_loader, model, criterion)

        end_time = time.time()

        epoch_time = end_time - start_time
        total_time = end_time - zero_time

        logging.info(
            'Total time used: %s Epoch %d time used: %s train loss: %.4f val loss: %.4f lr: %.6f'
            'valid_precision: %.3f valid_recall: %.3f valid_accuracy: %.3f valid_f1: %.3f' % (
                str(datetime.timedelta(seconds=int(total_time))), epoch,
                str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, lr_schedule.get_lr(),
                val_metric['precision'], val_metric['recall'], val_metric['accuracy'], val_metric['f1']))


def train(train_loader, model, criterion, optimizer, lr_schedule):
    model.train()
    total_loss = 0
    total_num_tokens = 0
    for input, label, mask in train_loader:
        input, label, mask = input.cuda(), label.cuda(), mask.cuda()
        output = model(input_ids=input, attention_mask=mask)
        loss, num_token = criterion(output, label, mask)
        optimizer.zero_grad()
        loss.backward()
        lr_schedule.step()
        optimizer.step()
        # Keep track of metrics
        total_loss += loss.item()
        total_num_tokens += num_token.item()

    return total_loss / total_num_tokens


def validate(valid_loader, model, criterion):
    model.eval()
    total_loss = 0
    total_num_tokens = 0
    total_predictions = []
    total_label = []
    total_mask = []
    for input, label, mask in valid_loader:
        input, label, mask = input.cuda(), label.cuda(), mask.cuda()
        output = model(input_ids=input, attention_mask=mask)
        loss, num_token = criterion(output, label, mask)
        total_predictions.extend(output.max(dim=-1)[1].tolist())
        total_label.extend(label.tolist())
        total_mask.extend(mask.tolist())
        total_loss += loss.item()
        total_num_tokens += num_token.item()
    return total_loss / total_num_tokens, compute_metrics(total_predictions, total_label, total_mask)


def test():
    ...


if __name__ == "__main__":
    with open("./configs.yaml", 'r') as fin:
        args = Arguments(yaml.load(fin, Loader=yaml.FullLoader))
    print(args.__dict__)
    main(args)
