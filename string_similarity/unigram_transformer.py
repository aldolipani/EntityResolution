import math
import pickle

import torch
import torch.nn as nn

from string_similarity import EntitySimilarity


class PositionalEncoding:

    def __init__(self, embedding_dim, max_len=100):
        self.pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def encode(self, x):
        return x + self.pe


class Model(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 nhead=8, num_layers=2, dim_feedforward=1024, max_len=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, activation='gelu'), num_layers)
        self.fc1 = nn.Linear(embedding_dim * max_len, embedding_dim >> 2)
        self.fc2 = nn.Linear(embedding_dim >> 2, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.embed(x)
        y_pred = self.pos_encoder.encode(y_pred) * math.sqrt(self.embedding_dim)
        y_pred = self.transformer_encoder(y_pred)
        y_pred = y_pred.reshape((y_pred.shape[0], -1))
        y_pred = self.fc1(y_pred)
        y_pred = self.fc2(y_pred)
        if y is None:
            y_pred = nn.functional.softmax(y_pred, dim=0)[:, 1]
            return y_pred
        else:
            return self.criterion(y_pred, y), y_pred


class UnigramTransformerSimilarity(EntitySimilarity):

    def __init__(self, max_len=100):
        super(UnigramTransformerSimilarity, self).__init__("Unigram Transformer")
        self.max_len = max_len
        self.character_to_idx = None
        self.model = None

    def load(self):
        with open("./models/character_to_idx.dat", "rb") as f:
            self.character_to_idx = pickle.load(f)
        self.model = Model(len(self.character_to_idx), 128, max_len=self.max_len)
        self.model.load_state_dict(torch.load('./models/unigram_transformer.model'))
        self.model.eval()

    def compute(self, entity1: str, entity2: str):

        pass

    def to_idxs(self, str_entity: str):
        res = []
        for character in str_entity:
            if character in self.character_to_idx:
                res.append(self.character_to_idx[character])
            else:
                res.append(self.character_to_idx['[unk]'])
        return res

    def add_padding(x):

        return x

    def preprocess(self, str_entity1, str_entity2):
        # from chars to ids
        x = [self.character_to_idx['[cls]']]
        x += self.to_idxs(str_entity1)
        x += [self.character_to_idx['[sep]']]
        x += self.to_idxs(str_entity2)
        # add crop or add padding if length does not mach
        if len(x) > self.max_len:
            x = x[0:self.max_len]
        else:
            x += [self.character_to_idx['[pad]']] * (self.max_len - len(x))
        return x
