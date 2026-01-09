# English (Roman Script) to Devanagari Transliteration

## Overview
This project implements a machine learning–based transliteration system that converts text written in English (Roman script) into its phonetic representation in Devanagari script.

This task focuses on transliteration, not translation. The system preserves pronunciation while converting the writing system.

---

## Problem Statement
Given an input text written in English (Romanized form), the system outputs the corresponding Devanagari representation.

### Example (Phonetic Transliteration)

| Input | Output |
|------|--------|
| Hello, how are you? | हैलो, हाउ आर यू? |

Note: The output reflects phonetic transliteration, not semantic translation.

---

## Dataset
- Dataset: Aksharantar (AI4Bharat)
- Language: Hindi
- Task: Roman script → Devanagari transliteration
- Size: ~1.3 million word pairs
- Format: JSONL (converted to TSV)

Each entry consists of:
romanized_word<TAB>devanagari_word

Example:
janamdivas    जन्म दिवस

---

## Data Preprocessing
- Converted JSONL files to TSV format
- Character-level tokenization
- Vocabulary construction for source and target scripts
- Added special tokens: <PAD>, <SOS>, <EOS>
- Filtered sequences exceeding maximum length

---

## Model Architecture
The system uses a character-level Seq2Seq LSTM model.

- Encoder: LSTM
- Decoder: LSTM
- Embedding Dimension: 64
- Hidden Dimension: 128
- Loss Function: Cross-Entropy Loss (padding ignored)
- Optimizer: Adam
- Training Strategy: Teacher forcing
- Decoding Strategy: Greedy decoding

Character-level modeling enables generalization to unseen words.

---

## Project Structure
.
├── data/
│   └── raw/
│       └── hi_train_sample.tsv
├── src/
│   ├── convert_jsonl_to_tsv.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── infer.py
├── requirements.txt
└── README.md

---

## Setup Instructions
1. Clone the repository:
   git clone
   cd english-to-devanagari-transliteration

2. Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

---

## Training
To train the model, run:
python src/train.py

The training script loads data, builds vocabularies, trains the model, and saves learned parameters locally.

---

## Inference
To transliterate a romanized input:
python src/infer.py namaste


## Notes on Sentence-Level Inputs
The model is trained primarily on word-level transliteration pairs. Sentence-level inputs containing spaces or punctuation may require additional preprocessing and were kept out of scope to focus on the core assignment requirements.

---

## Limitations and Future Work
- Greedy decoding may introduce minor spelling variations
- Accuracy can be improved by training on more data or epochs
- Future improvements include attention mechanisms, beam search, and Transformer-based models

---





