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

CFG = {
    'fold_num': 5,
    'seed': 42,
    'model': './data/roberta-base',
    'max_len': 512,
    'epochs': 3,
    'train_bs': 16,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 0,
    'weight_decay': 1e-5,
}


# 定义评价指标
def compute_metrics(predictions, labels, masks):
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[id2label[p] for (p, m) in zip(prediction, mask) if m] for prediction, mask in
                        zip(predictions, masks)]
    true_labels = [[id2label[l] for (l, m) in zip(labels, mask) if m] for labels, mask in
                   zip(labels, masks)]
    accuracy, precision, recall, f1 = [], [], [], []
    for prediction, label in zip(true_predictions, true_labels):
        accuracy.append(metrics.accuracy_score(y_pred=prediction, y_true=label))
        precision.append(metrics.precision_score(y_pred=prediction, y_true=label))
        recall.append(metrics.recall_score(y_pred=prediction, y_true=label))
        f1.append(metrics.f1_score(y_pred=prediction, y_true=label))
    return {"precision": sum(precision)/len(precision), "recall": sum(recall)/len(recall),
            "accuracy": sum(accuracy)/len(accuracy), "f1": sum(f1)/len(f1)}


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
                                 lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
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
                val_metric['precision'], val_metric['recall'], val_metric['accuracy'], val_metric['f1'] ))


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
    total_input = []
    total_label = []
    total_mask = []
    for input, label, mask in valid_loader:
        input, label, mask = input.cuda(), label.cuda(), mask.cuda()
        output = model(input_ids=input, attention_mask=mask)
        loss, num_token = criterion(output, label, mask)
        total_input.extend(input.to_numpy())
        total_label.extend(label.to_numpy())
        total_mask.extend(mask.to_numpy())
    return total_loss / total_num_tokens, compute_metrics(total_input, total_label, total_mask)


def test():
    ...


if __name__ == "__main__":
    with open("./configs.yaml", 'r') as fin:
        args = Arguments(yaml.load(fin, Loader=yaml.FullLoader))
    main(args)
