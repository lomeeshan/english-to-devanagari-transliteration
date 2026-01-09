import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_loader import (
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN,
    load_tsv, build_vocab, encode
)
from model import Encoder, Decoder, Seq2Seq

# ---------------------------
# Config
# ---------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"  
    subset_size: int = 500_000  
    max_src_len: int = 25
    max_tgt_len: int = 25

    batch_size: int = 64
    emb_dim: int = 64
    hidden_dim: int = 128

    lr: float = 1e-3
    epochs: int = 10

    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.5  # taper down slightly

    save_dir: str = "models"
    save_name: str = "seq2seq_lstm_hin.pt"
    vocab_name: str = "vocabs_hin.json"


# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

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
        max_src_len: int,
        max_tgt_len: int,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_src = src_vocab[PAD_TOKEN]
        self.pad_tgt = tgt_vocab[PAD_TOKEN]

        data = []
        for src, tgt in pairs:
            src_ids = encode(src, src_vocab)
            tgt_ids = encode(tgt, tgt_vocab)
            if len(src_ids) <= max_src_len and len(tgt_ids) <= max_tgt_len:
                data.append((src_ids, tgt_ids))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def make_collate_fn(pad_src: int, pad_tgt: int):
    def collate(batch):
        src_seqs = [b[0] for b in batch]
        tgt_seqs = [b[1] for b in batch]
        src = pad_batch(src_seqs, pad_src)
        tgt = pad_batch(tgt_seqs, pad_tgt)
        return src, tgt
    return collate

def schedule_teacher_forcing(cfg: TrainConfig, epoch_idx: int) -> float:
    # Linear schedule from start to end
    if cfg.epochs <= 1:
        return cfg.teacher_forcing_end
    frac = epoch_idx / (cfg.epochs - 1)
    return cfg.teacher_forcing_start + frac * (cfg.teacher_forcing_end - cfg.teacher_forcing_start)

def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {i: ch for ch, i in vocab.items()}

@torch.no_grad()
def quick_eval_examples(
    model: Seq2Seq,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    examples: List[str],
    device: torch.device,
    max_len: int = 30,
):
    inv_tgt = invert_vocab(tgt_vocab)

    def encode_src(s: str) -> torch.Tensor:
        ids = [src_vocab[SOS_TOKEN]] + [src_vocab.get(ch, src_vocab[PAD_TOKEN]) for ch in s] + [src_vocab[EOS_TOKEN]]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    for s in examples:
        src = encode_src(s).to(device)
        out = model.greedy_decode(src, max_len=max_len)[0].tolist()

        # Convert predicted IDs to chars, stop at EOS
        chars = []
        for tid in out[1:]:
            if tid == tgt_vocab[EOS_TOKEN]:
                break
            if tid == tgt_vocab[PAD_TOKEN]:
                continue
            chars.append(inv_tgt.get(tid, ""))
        pred = "".join(chars)
        print(f"  IN:  {s}")
        print(f"  OUT: {pred}")
        print("")

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---------------------------
    # Load data
    # ---------------------------
    train_path = Path("data/raw/hi_train.tsv")
    pairs = load_tsv(train_path)
    random.shuffle(pairs)
    pairs = pairs[: cfg.subset_size]

    src_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]

    src_vocab = build_vocab(src_texts)
    tgt_vocab = build_vocab(tgt_texts)

    dataset = TransliterationDataset(
        pairs, src_vocab, tgt_vocab,
        max_src_len=cfg.max_src_len,
        max_tgt_len=cfg.max_tgt_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(src_vocab[PAD_TOKEN], tgt_vocab[PAD_TOKEN]),
    )

    print("Pairs loaded:", len(pairs))
    print("After length filter:", len(dataset))
    print("SRC vocab:", len(src_vocab), "TGT vocab:", len(tgt_vocab))

    # ---------------------------
    # Build model
    # ---------------------------
    enc = Encoder(len(src_vocab), cfg.emb_dim, cfg.hidden_dim, pad_id=src_vocab[PAD_TOKEN])
    dec = Decoder(len(tgt_vocab), cfg.emb_dim, cfg.hidden_dim, pad_id=tgt_vocab[PAD_TOKEN])
    model = Seq2Seq(
        enc, dec,
        sos_id=tgt_vocab[SOS_TOKEN],
        eos_id=tgt_vocab[EOS_TOKEN],
        pad_id=tgt_vocab[PAD_TOKEN],
    ).to(device)

    # Loss ignores PAD tokens
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab[PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---------------------------
    # Train
    # ---------------------------
    model.train()
    for epoch in range(cfg.epochs):
        tf = schedule_teacher_forcing(cfg, epoch)
        total_loss = 0.0
        n_batches = 0

        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()

            # logits: (B, T-1, V) predicting tgt[:,1:]
            logits = model(src, tgt, teacher_forcing_ratio=tf)

            # reshape for CE: (B*(T-1), V) vs (B*(T-1))
            B, Tm1, V = logits.shape
            loss = criterion(logits.reshape(B * Tm1, V), tgt[:, 1:].reshape(B * Tm1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{cfg.epochs} | teacher_forcing={tf:.2f} | avg_loss={avg_loss:.4f}")

        # qualitative check each epoch
        model.eval()
        quick_eval_examples(
            model, src_vocab, tgt_vocab,
            examples=["janamdivas", "bharat", "dilli", "namaste"],
            device=device,
        )
        model.train()

    # ---------------------------
    # Save
    # ---------------------------
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / cfg.save_name
    torch.save(
        {
            "model_state": model.state_dict(),
            "emb_dim": cfg.emb_dim,
            "hidden_dim": cfg.hidden_dim,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
        },
        ckpt_path,
    )

    vocab_path = save_dir / cfg.vocab_name
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f, ensure_ascii=False, indent=2)

    print(f"Saved model checkpoint to: {ckpt_path}")
    print(f"Saved vocabs to: {vocab_path}")

if __name__ == "__main__":
    main()
