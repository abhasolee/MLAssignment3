import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import urllib.request
import re
from collections import Counter
import numpy as np
from gensim.models import Word2Vec
from datasets import load_dataset

class ShakespeareDataset(Dataset):
    def __init__(self, seq_length=20, max_vocab=5000, embed_size=128):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("Downloading Shakespeare text...")
        text = urllib.request.urlopen(url).read().decode('utf-8')
        
        words = re.findall(r'\w+|[^\w\s]', text)
        
        word_counts = Counter(words)
        top_words = [w for w, c in word_counts.most_common(max_vocab - 1)]
        self.vocab = ['<UNK>'] + top_words
        self.vocab_size = len(self.vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        
        self.encoded_text = [self.word2idx.get(w, 0) for w in words]
        self.seq_length = seq_length

        print("Training Word2Vec on the dataset...")
        sentences = [words[i:i+seq_length] for i in range(0, len(words), seq_length)]
        w2v_model = Word2Vec(sentences, vector_size=embed_size, window=5, min_count=1, workers=4)
        
        weights = np.zeros((self.vocab_size, embed_size))
        for i, word in enumerate(self.vocab):
            if word in w2v_model.wv:
                weights[i] = w2v_model.wv[word]
            else:
                weights[i] = np.random.normal(scale=0.6, size=(embed_size,))
        self.embedding_weights = torch.FloatTensor(weights)
        print("Word2Vec training complete.")

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.encoded_text[idx : idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class Multi30kDataset(Dataset):
    def __init__(self, max_vocab=3000, max_len=20, embed_size=128):
        print("Downloading Multi30k Dataset...")
        dataset = load_dataset("bentrevett/multi30k")
        train_data = dataset['train']
        
        self.en_texts = [s.lower().split() for s in train_data['en']]
        self.de_texts = [s.lower().split() for s in train_data['de']]
        
        self.en_texts = self.en_texts[:10000]
        self.de_texts = self.de_texts[:10000]
        self.max_len = max_len
        
        self.en_vocab, self.en_w2i, self.en_i2w = self.build_vocab(self.en_texts, max_vocab)
        self.de_vocab, self.de_w2i, self.de_i2w = self.build_vocab(self.de_texts, max_vocab)
        
        self.src_vocab_size = len(self.en_vocab)
        self.trg_vocab_size = len(self.de_vocab)
        
        print("Training English Word2Vec...")
        w2v_en = Word2Vec(self.en_texts, vector_size=embed_size, window=5, min_count=1, workers=4)
        self.src_weights = self.extract_weights(w2v_en, self.en_vocab, embed_size)
        
        print("Training German Word2Vec...")
        w2v_de = Word2Vec(self.de_texts, vector_size=embed_size, window=5, min_count=1, workers=4)
        self.trg_weights = self.extract_weights(w2v_de, self.de_vocab, embed_size)

    def build_vocab(self, texts, max_vocab):
        counter = Counter(word for sentence in texts for word in sentence)
        # 0: PAD, 1: SOS, 2: EOS, 3: UNK
        words = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [w for w, c in counter.most_common(max_vocab - 4)]
        word2idx = {w: i for i, w in enumerate(words)}
        idx2word = {i: w for i, w in enumerate(words)}
        return words, word2idx, idx2word

    def extract_weights(self, w2v_model, vocab, embed_size):
        weights = torch.zeros((len(vocab), embed_size))
        for i, word in enumerate(vocab):
            if word in w2v_model.wv:
                weights[i] = torch.FloatTensor(w2v_model.wv[word])
            else:
                weights[i] = torch.randn(embed_size) # Random init for special tokens
        return weights

    def encode(self, text, word2idx):
        tokens = ['<SOS>'] + text[:self.max_len-2] + ['<EOS>']
        indices = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
        pads = [word2idx['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices + pads, dtype=torch.long)

    def __len__(self):
        return len(self.en_texts)

    def __getitem__(self, idx):
        src = self.encode(self.en_texts[idx], self.en_w2i)
        trg = self.encode(self.de_texts[idx], self.de_w2i)
        return src, trg
    
def get_shakespeare_dataloader(seq_length=20, batch_size=128, max_vocab=5000):
    dataset = ShakespeareDataset(seq_length, max_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, dataset.vocab_size, dataset.embedding_weights

def get_task2_dataloader(batch_size=64, max_vocab=3000, max_len=20, embed_size=128):
    dataset = Multi30kDataset(max_vocab, max_len, embed_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, dataset