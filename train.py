import datetime
import time
import pandas as pd
import torch
import yaml
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LongformerConfig

from dataloader import load_data, FPDataset, num_labels, id2label, label2id, load_test_data
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
    precision = metrics.precision_score(y_pred=predictions, y_true=labels, average='micro')
    recall = metrics.recall_score(y_pred=predictions, y_true=labels, average='micro')
    f1 = metrics.f1_score(y_pred=predictions, y_true=labels, average='micro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def save_test(ids, inputs, preds, tokenizer, f):
    inputs = [tokenizer.convert_ids_to_tokens(sent) for sent in inputs]
    res = []
    for idx, sent, pred in zip(ids, inputs, preds):
        num = -1
        pre_idx = -1
        pre_label = "O"
        for x, (word, label_id) in enumerate(zip(sent[1:-1], pred[1:-1])):
            label = id2label[label_id]
            if x == 0 or word[0] == "Ġ":
                num += 1
            else:
                continue
            if pre_idx < 0:
                if label[:1] == "B":
                    pre_idx = num
                    pre_label = label[2:]
            else:
                if label[:1] == "I" and pre_label == label[2:]:
                    continue
                else:
                    res.append((idx, pre_label, " ".join([str(i) for i in range(pre_idx, num)])))
                    if label[:1] == "B":
                        pre_idx = num
                        pre_label = label[2:]
                    else:
                        pre_idx = -1
                        pre_label = "O"
    res = pd.DataFrame(res, columns=["id", "class", "predictionstring"])
    res.to_csv(f, index=False, encoding='utf8')


def main(args):
    cp_file = args.RESULT_DIR + '/' + "best_model.pt"
    result_file = args.RESULT_DIR + '/' + "test_result.csv"
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    train_data, valid_data = load_data(tokenizer)
    test_data = load_test_data(tokenizer)
    train_dataset = FPDataset(train_data, tokenizer, batch_size=args.batch_size)
    valid_dataset = FPDataset(valid_data, tokenizer, batch_size=args.batch_size)
    test_dataset = FPDataset(test_data, tokenizer, batch_size=args.batch_size)
    train_loader = DataLoader(dataset=train_dataset, batch_size=None, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=None)
    config = LongformerConfig.from_pretrained("allenai/longformer-base-4096", num_labels=num_labels)
    model = LongformerSoftmaxForNer.from_pretrained("allenai/longformer-base-4096", config=config).cuda()
    criterion = LabelSmoothedCrossEntropyCriterion(eps=args.label_smoothing)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, betas=(args.betas0, args.betas1))
    lr_schedule = InverseSquareRootSchedule(optimizer, lr=args.lr, warmup_updates=args.warmup_updates)
    best_val_f1 = 0
    zero_time = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train(train_loader, model, criterion, optimizer, lr_schedule)
        val_loss, val_metric = validate(valid_loader, model, criterion)

        end_time = time.time()

        epoch_time = end_time - start_time
        total_time = end_time - zero_time

        print('Total time used: %s Epoch %d time used: %s train loss: %.4f val loss: %.4f '
              'valid_accuracy: %.3f valid_precision: %.3f valid_recall: %.3f valid_f1: %.3f' % (
                  str(datetime.timedelta(seconds=int(total_time))), epoch,
                  str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss,
                  val_metric['accuracy'], val_metric['precision'], val_metric['recall'], val_metric['f1']))
        val_f1 = val_metric['f1']
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        }, cp_file)
    print("load best model!!!")
    model.load_state_dict(torch.load(cp_file)['state_dict'])
    ids, inputs, predictions = test(test_loader, model)
    save_test(ids, inputs, predictions, tokenizer=tokenizer, f=result_file)


def train(train_loader, model, criterion, optimizer, lr_schedule):
    model.train()
    total_loss = 0
    total_num_tokens = 0
    for _, inputs, label, mask in train_loader:
        inputs, label, mask = inputs.cuda(), label.cuda(), mask.cuda()
        output = model(input_ids=inputs, attention_mask=mask)
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
    with torch.no_grad():
        for _, inputs, label, mask in valid_loader:
            inputs, label, mask = inputs.cuda(), label.cuda(), mask.cuda()
            output = model(input_ids=inputs, attention_mask=mask)
            loss, num_token = criterion(output, label, mask)
            total_predictions.extend(output.max(dim=-1)[1].tolist())
            total_label.extend(label.tolist())
            total_mask.extend(mask.tolist())
            total_loss += loss.item()
            total_num_tokens += num_token.item()
    return total_loss / total_num_tokens, compute_metrics(total_predictions, total_label, total_mask)


def test(test_loader, model):
    model.eval()
    total_inputs = []
    predictions = []
    masks = []
    ids = []
    with torch.no_grad():
        for idx, inputs, _, mask in test_loader:
            inputs, mask = inputs.cuda(), mask.cuda()
            output = model(input_ids=inputs, attention_mask=mask)
            total_inputs.extend(inputs.tolist())
            predictions.extend(output.max(dim=-1)[1].tolist())
            masks.extend(mask.tolist())
            ids.extend(idx)
    inputs = [[i for (i, m) in zip(inputs, mask) if m] for inputs, mask in zip(total_inputs, masks)]
    predictions = [[p for (p, m) in zip(prediction, mask) if m] for prediction, mask in zip(predictions, masks)]
    return ids, inputs, predictions


if __name__ == "__main__":
    with open("./configs.yaml", 'r') as fin:
        args = Arguments(yaml.load(fin, Loader=yaml.FullLoader))
    print(args.__dict__)
    main(args)
