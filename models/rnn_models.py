import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, rnn_type='LSTM', use_one_hot=False, pretrained_embeddings=None):
        super(TextGenerator, self).__init__()
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        
        if use_one_hot:
            self.embedding = None 
            embed_size = vocab_size 
        else:
            embed_size = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        if self.use_one_hot:
            x = F.one_hot(x, num_classes=self.vocab_size).float()
        else:
            x = self.embedding(x)
            
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden