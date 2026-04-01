import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', use_one_hot=False, pretrained_embeddings=None):
        super(Encoder, self).__init__()
        self.use_one_hot = use_one_hot
        self.vocab_size = input_size
        self.rnn_type = rnn_type
        
        if use_one_hot:
            self.embedding = None
            embed_size = input_size
        else:
            embed_size = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        if self.use_one_hot:
            embedded = F.one_hot(x, num_classes=self.vocab_size).float()
        else:
            embedded = self.embedding(x)
            
        if self.rnn_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
            return hidden, cell
        else:
            outputs, hidden = self.rnn(embedded)
            return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, rnn_type='LSTM', use_one_hot=False, pretrained_embeddings=None):
        super(Decoder, self).__init__()
        self.use_one_hot = use_one_hot
        self.vocab_size = output_size
        self.rnn_type = rnn_type
        
        if use_one_hot:
            self.embedding = None
            embed_size = output_size
        else:
            embed_size = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell=None):
        x = x.unsqueeze(1)
        
        if self.use_one_hot:
            embedded = F.one_hot(x, num_classes=self.vocab_size).float()
        else:
            embedded = self.embedding(x)
        
        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            return prediction, hidden, cell
        else:
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc(output.squeeze(1))
            return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.rnn_type = encoder.rnn_type

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        if self.rnn_type == 'LSTM':
            hidden, cell = self.encoder(src)
        else:
            hidden = self.encoder(src)
            cell = None
            
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            if self.rnn_type == 'LSTM':
                output, hidden, cell = self.decoder(input, hidden, cell)
            else:
                output, hidden = self.decoder(input, hidden)
                
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs