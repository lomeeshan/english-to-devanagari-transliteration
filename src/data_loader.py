from pathlib import Path
from typing import List, Tuple, Dict
import random

import torch
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

def load_tsv(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line.split("\t")
            pairs.append((src, tgt))
    return pairs

def build_vocab(sequences: List[str]) -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
    idx = 3
    for seq in sequences:
        for ch in seq:
            if ch not in vocab:
                vocab[ch] = idx
                idx += 1
    return vocab

def encode(seq: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[SOS_TOKEN]] + [vocab[ch] for ch in seq] + [vocab[EOS_TOKEN]]

def pad_batch(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out

class TransliterationDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        max_src_len: int = 25,
        max_tgt_len: int = 25,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_id_src = src_vocab[PAD_TOKEN]
        self.pad_id_tgt = tgt_vocab[PAD_TOKEN]

        filtered = []
        for src, tgt in pairs:
            src_ids = encode(src, src_vocab)
            tgt_ids = encode(tgt, tgt_vocab)
            if len(src_ids) <= max_src_len and len(tgt_ids) <= max_tgt_len:
                filtered.append((src_ids, tgt_ids))

        self.data = filtered

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

def make_collate_fn(pad_id_src: int, pad_id_tgt: int):
    def collate(batch):
        src_seqs = [x[0] for x in batch]
        tgt_seqs = [x[1] for x in batch]
        src = pad_batch(src_seqs, pad_id_src)
        tgt = pad_batch(tgt_seqs, pad_id_tgt)
        return src, tgt
    return collate

if __name__ == "__main__":
    random.seed(42)

    train_path = Path("data/raw/hi_train.tsv")
    pairs = load_tsv(train_path)

    # Shuffle and take a manageable subset for experimentation
    random.shuffle(pairs)
    pairs_small = pairs[:200000]  # 200k is plenty to start fast

    src_texts = [p[0] for p in pairs_small]
    tgt_texts = [p[1] for p in pairs_small]

    src_vocab = build_vocab(src_texts)
    tgt_vocab = build_vocab(tgt_texts)

    dataset = TransliterationDataset(
        pairs_small, src_vocab, tgt_vocab, max_src_len=25, max_tgt_len=25
    )

    print("Original pairs:", len(pairs))
    print("Subset pairs:", len(pairs_small))
    print("After length filter:", len(dataset))
    print("SRC vocab:", len(src_vocab), "TGT vocab:", len(tgt_vocab))

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=make_collate_fn(src_vocab[PAD_TOKEN], tgt_vocab[PAD_TOKEN]),
    )

    src_batch, tgt_batch = next(iter(loader))
    print("SRC batch shape:", tuple(src_batch.shape))
    print("TGT batch shape:", tuple(tgt_batch.shape))

