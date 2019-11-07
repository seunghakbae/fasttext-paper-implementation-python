import pandas as pd
import json
import nltk
from pathlib import Path

datapath = Path('/data/ag_news_csv')
test_csv = datapath / 'test_csv'
train_csv = datapath / 'train_csv'


def tokenize(txt):
    tokens = [word for sentence in nltk.sent_tokenize(txt)
              for word in nltk.word_tokenize(sentence)]
    # print(type(tokens))
    return tokens


def bigram(tokens):
    bigram = []

    for i in range(len(tokens) - 1):
        bigram.append(tokens[i] + " " + tokens[i + 1])
    return bigram


datapath = Path('data/ag_news_csv')

test_data = pd.read_csv(datapath / 'test.csv', names=["class", "title", "body"])
train_data = pd.read_csv(datapath / 'train.csv', names=["class", "title", "body"])

print(test_data)
print(train_data)