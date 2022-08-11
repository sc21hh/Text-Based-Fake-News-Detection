import os
from collections import Counter
from data_process import get_dataframe
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import nltk


class Vocabulary(object):
    """ Simple vocabulary wrapper which maps every unique word to an integer ID. """

    def __init__(self):
        # intially, set both the IDs and words to dictionaries with special tokens
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<end>': 2}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<end>'}
        self.idx = 3

    def add_word(self, word):
        # if the word does not already exist in the dictionary, add it
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            # increment the ID for the next word
            self.idx += 1

    def __call__(self, word):
        # if we try to access a word not in the dictionary, return the id for <unk>
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class MyDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab

    def __getitem__(self, index):

        text = self.texts[index]
        token = nltk.tokenize.word_tokenize(text)
        text_vec = [self.vocab('<end>') if i == len(token) else (self.vocab(token[i]) if len(token) > i else self.vocab('<pad>')) for i
                    in range(760)]
        return torch.tensor(text_vec), self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    import configparser, sys

    config_path = os.path.join(os.path.split(os.path.split(os.path.realpath(sys.argv[0]))[0])[0], 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    root_path = config['config']['ROOT_PATH']

    df = get_dataframe(root_path)

    rows, cols = df.shape
    split_index_1 = int(rows * 0.2)
    split_index_2 = int(rows * 0.4)
    # 数据分割
    test_df = df.iloc[0: split_index_1, :]
    val_df = df.iloc[split_index_1:split_index_2, :]
    train_df = df.iloc[split_index_2: rows, :]

    test_df.to_hdf(os.path.join(root_path, 'dataset/test_df.h5'), key='df', mode='w')
    val_df.to_hdf(os.path.join(root_path, 'dataset/val_df.h5'), key='df', mode='w')
    train_df.to_hdf(os.path.join(root_path, 'dataset/train_df.h5'), key='df', mode='w')

    counter = Counter()

    for text in train_df['text'].tolist():
        token = nltk.tokenize.word_tokenize(text)
        counter.update(token)

    for text in val_df['text'].tolist():
        token = nltk.tokenize.word_tokenize(text)
        counter.update(token)

    words = [word for word, count in counter.items() if count >= 4]
    print(len(words))  # 55783

    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    import pickle

    with open(os.path.join(root_path, 'dataset/vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    train_set = MyDataset(train_df, vocab)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    train_iter = iter(train_loader)
    texts, labels = train_iter.next()
    print(texts.shape)  # torch.Size([64, 430])
    print(labels.shape)  # torch.Size([64])
