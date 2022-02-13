import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer

LABELS = ['O', 'Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement']
id2label = ['O', 'B-Lead', 'I-Lead', 'B-Position', "I-Position", 'B-Claim', 'I-Claim',
            'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal',
            'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
label2id = {tag: idx for idx, tag in enumerate(id2label)}
num_labels = len(id2label)


def load_data(tokenizer):
    data = pd.read_csv('data/train.csv', encoding='utf8')[["id", "discourse_text", "discourse_type"]]
    data.discourse_text = data.discourse_text.map(
        lambda x: " ".join(x.split()))
    data_names, data_texts = [], []
    for f in tqdm(list(os.listdir('data/train'))):
        data_names.append(f.replace('.txt', ''))
        data_texts.append(open('data/train/' + f, 'r', encoding="utf8").read())
    data_texts = pd.DataFrame({'id': data_names, 'text': data_texts})
    data_texts.text = data_texts.text.map(lambda x: " ".join(x.split()))
    data = pd.merge(data_texts, data, how='left', on=['id'])
    data['label'] = data.apply(lambda row: [row['text'].find(row['discourse_text']),
                                            row['text'].find(row['discourse_text']) +
                                            len(row['discourse_text'])], axis=1)
    data['label'] = data['label'] + data['discourse_type'].map(lambda x: [x])
    data = data.groupby("id").agg(text=("text", lambda x: x.iloc[0]), label=("label", list)).reset_index()
    data['label'] = data.apply(handleLabel, args=(tokenizer,), axis=1)
    data['text'] = data['text'].apply(lambda x: tokenizer(x)["input_ids"])
    return data[:13000], data[13000:]


def load_test_data(tokenizer):
    data_names, data_texts = [], []
    for f in tqdm(list(os.listdir('data/test'))):
        data_names.append(f.replace('.txt', ''))
        data_texts.append(open('data/test/' + f, 'r', encoding="utf8").read())
    data = pd.DataFrame({'id': data_names, 'text': data_texts})
    data.text = data.text.map(lambda x: " ".join(x.split()))
    data['label'] = data.apply(lambda row: [], axis=1)
    data['label'] = data.apply(handleLabel, args=(tokenizer,), axis=1)
    data['text'] = data['text'].apply(lambda x: tokenizer(x)["input_ids"])
    return data


def handleLabel(row, tokenizer):
    nlabels = []
    text = row["text"]
    labels = sorted(row["label"], key=lambda x: x[0])
    offset = tokenizer(text, return_offsets_mapping=True)["offset_mapping"][1:-1]
    i = idx = 0
    pre_idx = -1
    nlabels.append(label2id['O'])
    while i < len(offset):
        s, e = offset[i]
        if idx == len(labels):
            nlabels.append(label2id['O'])
        elif labels[idx][0] <= s < labels[idx][1] or labels[idx][0] < e <= labels[idx][1]:
            if pre_idx == idx:
                tlabel = "I-" + labels[idx][-1]
            else:
                tlabel = "B-" + labels[idx][-1]
            pre_idx = idx
            nlabels.append(label2id[tlabel])
        elif labels[idx][1] <= s:
            idx += 1
            continue
        elif labels[idx][0] >= e:
            nlabels.append(label2id['O'])
        i += 1
    nlabels.append(label2id['O'])
    return nlabels


def collate_wrapper(batch, padding):
    batch = batch[0]
    max_len = 0
    for input, label in batch:
        max_len = max(len(input), max_len)
    batch_input, batch_label = [], []
    for input, label in batch:
        batch_input.append(input + [padding] * (max_len - len(input)))
        batch_label.append(label + [padding] * (max_len - len(label)))
    batch_input = torch.LongTensor(batch_input)
    batch_label = torch.LongTensor(batch_label)
    batch_mask = batch_input != padding
    return batch_input, batch_label, batch_mask


class FPDataset(Dataset):

    def __init__(self, data, tokenizer, batch_size=4096):
        self.tokenizer = tokenizer
        data = sorted(data.to_numpy(), key=lambda x: len(x[1]))
        self.iter_data = []
        batch_idx = []
        batch_input = []
        batch_label = []
        max_len = 0
        for idx, text, tag in data:
            if (len(batch_input) + 1) * max(len(text), max_len) <= batch_size:
                max_len = max(len(text), max_len)
                batch_input.append(text)
                batch_label.append(tag)
                batch_idx.append(idx)
            else:
                self.iter_data.append(
                    (batch_idx, [t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in batch_input],
                     [l + [tokenizer.pad_token_id] * (max_len - len(l)) for l in batch_label]))
                max_len = len(text)
                batch_input = [text]
                batch_label = [tag]
                batch_idx = [idx]
        self.iter_data.append((batch_idx, [t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in batch_input],
                               [l + [tokenizer.pad_token_id] * (max_len - len(l)) for l in batch_label]))

    def __len__(self):
        return len(self.iter_data)

    def __getitem__(self, item):
        batch_idx, batch_input, batch_label = self.iter_data[item]
        batch_input, batch_label = torch.LongTensor(batch_input), torch.LongTensor(batch_label)
        batch_mask = batch_input != self.tokenizer.pad_token_id
        return batch_idx, batch_input, batch_label, batch_mask


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    train_data, valid_data = load_data(tokenizer)
    train_dataset = FPDataset(train_data, tokenizer)
    valid_dataset = FPDataset(valid_data, tokenizer)
    train_loader = DataLoader(dataset=train_dataset, batch_size=None, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=None)
