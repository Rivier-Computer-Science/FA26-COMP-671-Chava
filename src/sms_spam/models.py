"""Character-level text encoding and small TextCNN."""

from __future__ import annotations

import string

import torch
from torch import nn

CHAR_TO_ID = {ch: index + 2 for index, ch in enumerate(string.printable)}
VOCAB_SIZE = len(CHAR_TO_ID) + 2


def encode_message(message: str, max_length: int) -> list[int]:
    text = message.casefold().replace("\n", " ").replace("\r", " ")
    ids = [CHAR_TO_ID.get(ch, 1) for ch in text[:max_length]]
    return ids + [0] * (max_length - len(ids))


class TextCNN(nn.Module):
    def __init__(self, embedding_dim=16, filters_per_kernel=24,
                 kernel_sizes=(3, 5), dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters_per_kernel, kernel, padding=kernel // 2)
            for kernel in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(filters_per_kernel * len(kernel_sizes), 1)

    def forward(self, token_ids):
        lengths = token_ids.ne(0).sum(dim=1).clamp_min(1)
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convolutions:
            activation = torch.relu(conv(embedded))
            pos = torch.arange(activation.shape[-1], device=token_ids.device).unsqueeze(0)
            mask = pos < lengths.unsqueeze(1)
            activation = activation.masked_fill(~mask.unsqueeze(1), torch.finfo(activation.dtype).min)
            pooled.append(activation.max(dim=2).values)
        return self.output(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)
