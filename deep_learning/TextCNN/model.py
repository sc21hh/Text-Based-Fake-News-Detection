import torch
from torch import nn
from torch.nn import functional


class textCNN(nn.Module):

    def __init__(self, embedding_dim, vocab_size, out_dim, kernel_wins, num_class):
        super(textCNN, self).__init__()
        # Load pre-trained embeddings in the embedding layer.
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, out_dim, (w, embedding_dim)) for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(0.6)
        # FC layer
        self.fc = nn.Linear(len(kernel_wins) * out_dim, num_class)

    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)
        conv_x = [conv(emb_x) for conv in self.convs]
        pool_x = [functional.max_pool1d(x.squeeze(-1), x.size()[2]) for x in conv_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        output = self.fc(fc_x)
        return output
