import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from model import Encoder, Decoder, Seq2Seq
from data_loader import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {i: ch for ch, i in vocab.items()}

def encode_src(text: str, src_vocab: Dict[str, int]) -> torch.Tensor:
    # Unknown chars -> PAD (rare for roman)
    ids = [src_vocab[SOS_TOKEN]] + [src_vocab.get(ch, src_vocab[PAD_TOKEN]) for ch in text] + [src_vocab[EOS_TOKEN]]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, S)

def decode_tokens(token_ids: List[int], tgt_vocab: Dict[str, int], inv_tgt: Dict[int, str]) -> str:
    chars = []
    for tid in token_ids:
        if tid == tgt_vocab[SOS_TOKEN]:
            continue
        if tid == tgt_vocab[EOS_TOKEN]:
            break
        if tid == tgt_vocab[PAD_TOKEN]:
            continue
        chars.append(inv_tgt.get(tid, ""))
    return "".join(chars)

def main():
    parser = argparse.ArgumentParser(description="English(romanized) -> Devanagari transliteration (Hindi)")
    parser.add_argument("text", type=str, help="Input romanized word, e.g., janamdivas")
    parser.add_argument("--ckpt", type=str, default="models/seq2seq_lstm_hin.pt", help="Path to model checkpoint")
    parser.add_argument("--max_len", type=int, default=30, help="Max decode length")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location=device)

    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    emb_dim = ckpt["emb_dim"]
    hidden_dim = ckpt["hidden_dim"]

    enc = Encoder(len(src_vocab), emb_dim, hidden_dim, pad_id=src_vocab[PAD_TOKEN])
    dec = Decoder(len(tgt_vocab), emb_dim, hidden_dim, pad_id=tgt_vocab[PAD_TOKEN])
    model = Seq2Seq(enc, dec, sos_id=tgt_vocab[SOS_TOKEN], eos_id=tgt_vocab[EOS_TOKEN], pad_id=tgt_vocab[PAD_TOKEN])

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    inv_tgt = invert_vocab(tgt_vocab)

    src = encode_src(args.text.strip(), src_vocab).to(device)
    out = model.greedy_decode(src, max_len=args.max_len)[0].tolist()
    pred = decode_tokens(out, tgt_vocab, inv_tgt)

    print(pred)

if __name__ == "__main__":
    main()
