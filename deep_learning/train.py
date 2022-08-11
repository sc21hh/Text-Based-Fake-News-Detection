import os
import time
from collections import Counter
import pickle
import nltk
import numpy as np
import pandas as pd
import torch
from torch import nn

from deep_learning.LSTM.model import LSTM
from deep_learning.TextCNN.model import textCNN
from deep_learning.dataset import Vocabulary, MyDataset


def train(model, train_loader, val_loader, loss_fn, optimizer, path, filename):
    statsrec = np.zeros((4, nepochs))
    model = model.cuda()
    highest_acc = 0.25
    lowest_loss = 2.5

    for epoch in range(nepochs):

        # training data
        model.train()
        training_loss = 0.0
        n = 0
        correct = 0
        total = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        tra_acc = correct / total

        # validate
        model.eval()
        m = 0
        correct = 0
        total = 0
        validation_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                validation_loss += loss_fn(outputs, labels).item()
                m += 1
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = correct / total
        print(
            f"epoch: {epoch} cha: {training_loss / n - validation_loss / m: .5f} training loss: {training_loss / n: .3f} training accuracy: {tra_acc: .1%}  validation loss: {validation_loss / m} validation accuracy: {val_acc: .1%}")

        statsrec[:, epoch] = (training_loss / n, tra_acc, validation_loss / m, val_acc)

        if validation_loss / m < lowest_loss or val_acc > highest_acc:
            lowest_loss = validation_loss / m
            highest_acc = val_acc
            results_path = f'{path}/results/{filename}_{epoch}_loss_{lowest_loss: .3f}_acc_{highest_acc: .1%}.pt'
            torch.save({"state_dict": model.state_dict(), "stats": statsrec}, results_path)

    torch.save({"state_dict": model.state_dict(), "stats": statsrec}, f"{path}/results/{filename}.pt")


if __name__ == "__main__":
    import configparser, sys

    config_path = os.path.join(os.path.split(os.path.split(os.path.realpath(sys.argv[0]))[0])[0], 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    root_path = config['config']['ROOT_PATH']

    test_df = pd.read_hdf(os.path.join(root_path, 'dataset/test_df.h5'), 'df')
    val_df = pd.read_hdf(os.path.join(root_path, 'dataset/val_df.h5'), 'df')
    train_df = pd.read_hdf(os.path.join(root_path, 'dataset/train_df.h5'), 'df')

    if not os.path.isfile(os.path.join(root_path, 'dataset/vocab.pkl')):
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

        with open(os.path.join(root_path, 'dataset/vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open(os.path.join(root_path, 'dataset/vocab.pkl'), 'rb') as f:
            vocab = pickle.load(f)

    train_set = MyDataset(train_df, vocab)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    val_set = MyDataset(val_df, vocab)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=64, shuffle=True)

    nepochs = 100
    # net = LSTM(embedding_dim=100, hidden_dim=256, vocab_size=len(vocab), num_layers=2, n_class=2, bidirectional=True)
    net = textCNN(vocab_size=len(vocab), embedding_dim=300, out_dim=100, kernel_wins=[3, 4, 5], num_class=2)

    loss_fn = nn.CrossEntropyLoss()
    # # optimizer = torch.optim.SGD(single_training_net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00003)

    # train(net, train_loader, val_loader, "./LSTM", "LSTM")
    train(net, train_loader, val_loader, loss_fn, optimizer, "./TextCNN", "TextCNN")

    print('_----------------------------------_')
    time.sleep(300)
    lstm = LSTM(embedding_dim=100, hidden_dim=256, vocab_size=len(vocab), num_layers=2, n_class=2, bidirectional=True)
    loss_fn1 = nn.CrossEntropyLoss()
    # # optimizer = torch.optim.SGD(single_training_net.parameters(), lr=0.001, momentum=0.9)
    optimizer1 = torch.optim.Adam(lstm.parameters(), lr=0.00003)
    train(lstm, train_loader, val_loader, loss_fn1, optimizer1, "./LSTM", "LSTM")


