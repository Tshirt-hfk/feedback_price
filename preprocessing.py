import os
import pandas as pd
from tqdm import tqdm

LABELS = ['O', 'Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement']


def word_offset(text):
    offset = []
    pre = 0
    for pos, char in enumerate(text):
        if char == " ":
            if pos > pre:
                offset.append((pre, pos))
            pre = pos + 1
    if len(text) > pre:
        offset.append((pre, len(text)))
    return offset


def convert_label_offset(words, labels):
    res = []
    pre_pos = pos = 0
    pre_tag = "O"
    for i, (word, tag) in enumerate(zip(words, labels)):
        if tag != pre_tag:
            if pre_tag != "O":
                res.append((pre_pos, pos, pre_tag))
            pre_pos = pos + 1 if i > 0 else pos
        pos = pos + len(word) + 1 if i > 0 else pos + len(word)
        pre_tag = tag
    if pre_tag != "O":
        res.append((pre_pos, pos, pre_tag))
    return res


def split_sent(row, max_count=400):
    nlabels = []
    text = row["text"]
    labels = sorted(row["label"], key=lambda x: x[0])
    offset = word_offset(text)

    i = idx = 0
    pre_idx = -1
    while i < len(offset):
        s, e = offset[i]
        if idx == len(labels):
            nlabels.append('O')
        elif labels[idx][0] <= s < labels[idx][1] or labels[idx][0] < e <= labels[idx][1]:
            if pre_idx == idx:
                tlabel = labels[idx][-1]
            else:
                tlabel = labels[idx][-1]
            pre_idx = idx
            nlabels.append(tlabel)
        elif labels[idx][1] <= s:
            idx += 1
            continue
        elif labels[idx][0] >= e:
            nlabels.append('O')
        i += 1
    labels = nlabels
    words = text.split()
    assert len(labels) == len(words)

    paragraph = []
    paragraph_labels = []
    sent = []
    sent_labels = []
    pre_tag = "O"
    for word, tag in zip(words, labels):
        if tag != "O" and tag == pre_tag:
            sent.append(word)
            sent_labels.append(tag)
        elif word == ".":
            sent.append(word)
            sent_labels.append(tag)
            paragraph.append(sent)
            paragraph_labels.append(sent_labels)
            sent = []
            sent_labels = []
        else:
            sent.append(word)
            sent_labels.append(tag)
        pre_tag = tag
    res = []
    tmp_sent = []
    tmp_labels = []
    for sent, sent_labels in zip(paragraph, paragraph_labels):
        if len(tmp_sent) + len(sent) < max_count:
            tmp_sent.extend(sent)
            tmp_labels.extend(sent_labels)
        else:
            res.append((" ".join(tmp_sent), convert_label_offset(tmp_sent, tmp_labels)))
            tmp_sent = sent
            tmp_labels = sent_labels


data = pd.read_csv('data/feedback-prize/train.csv', encoding='utf8')[["id", "discourse_text", "discourse_type"]]
data.discourse_text = data.discourse_text.map(
    lambda x: " ".join(x.replace(",", " , ").replace(".", " . ").replace('"', ' " ')
                       .replace('!', ' ! ').replace('?', ' ? ').split()))
data_names, data_texts = [], []
for f in tqdm(list(os.listdir('./data/feedback-prize/train'))):
    data_names.append(f.replace('.txt', ''))
    data_texts.append(open('./data/feedback-prize/train/' + f, 'r', encoding="utf8").read())
data_texts = pd.DataFrame({'id': data_names, 'text': data_texts})
data_texts.text = data_texts.text.map(lambda x: " ".join(x.replace(",", " , ").replace(".", " . ")
                                                         .replace('"', ' " ').replace('!', ' ! ')
                                                         .replace('?', ' ? ').split()))
data = pd.merge(data_texts, data, how='left', on=['id'])
data['label'] = data.apply(lambda row: [row['text'].find(row['discourse_text']),
                                        row['text'].find(row['discourse_text']) +
                                        len(row['discourse_text'])], axis=1)
data['label'] = data['label'] + data['discourse_type'].map(lambda x: [x])
data = data.groupby("id").agg(text=("text", lambda x: x.iloc[0]), label=("label", list)).reset_index()

# data["sents"] = data.apply(split_sent, axis=1)
