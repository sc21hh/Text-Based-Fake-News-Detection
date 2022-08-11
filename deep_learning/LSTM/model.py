import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, n_class, bidirectional):
        super(LSTM, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=0.5)
        if bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        # [batch, seq_len] = > [batch, seq_len, embed_dim]
        emb_x = self.embed(x)
        # [batch, seq_len, embed_dim] => [seq_len, batch, embed_dim]
        states, hidden = self.rnn(emb_x.permute([1, 0, 2]))
        # states.shape= torch.Size([65, 64, 200])
        encoding = torch.cat([states[0], states[-1]], dim=1)
        # encoding.shape= torch.Size([64, 400])
        # decode
        outputs = self.decoder1(encoding)
        out = self.decoder2(outputs)
        return out


