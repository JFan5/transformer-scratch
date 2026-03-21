import os
import tempfile
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import sentencepiece as spm

def build_vocab(sentences):
      vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
      for sent in sentences:
          for word in sent.split():
              if word not in vocab:
                  vocab[word] = len(vocab)
      return vocab



# make the sentence of words to sentence of numbers, and add start and end token, for example, "hello world" -> [1, 4, 5, 2]   1 is <start>, 2 is <end>, 4 is "hello", 5 is "world"
def encode(sentence, vocab):
    tokens = [vocab["<start>"]]
    for word in sentence.split():
        tokens.append(vocab.get(word, vocab["<unk>"]))
    tokens.append(vocab["<end>"])
    return tokens


def pad_sequence(sequences):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        padded_seq = seq + [0] * (max_len - len(seq))
        padded_seqs.append(padded_seq)
    return padded_seqs


class TranslationDataset(Dataset):
    def __init__(self, en_sentences, de_sentences, en_vocab, de_vocab):
        self.en_sentences = en_sentences
        self.de_sentences = de_sentences
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sentence = self.en_sentences[idx]
        de_sentence = self.de_sentences[idx]
        en_tokens = encode(en_sentence, self.en_vocab)
        de_tokens = encode(de_sentence, self.de_vocab)
        return en_tokens, de_tokens


class BPETranslationDataset(Dataset):
    def __init__(self, en_sentences, de_sentences, en_sp, de_sp):
        self.en_sentences = en_sentences
        self.de_sentences = de_sentences
        self.en_sp = en_sp
        self.de_sp = de_sp

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        # bos_id=1 maps to <s> (start), eos_id=2 maps to </s> (end)
        en_tokens = [self.en_sp.bos_id()] + self.en_sp.encode(self.en_sentences[idx]) + [self.en_sp.eos_id()]
        de_tokens = [self.de_sp.bos_id()] + self.de_sp.encode(self.de_sentences[idx]) + [self.de_sp.eos_id()]
        return en_tokens, de_tokens


def train_sentencepiece(sentences, model_prefix, vocab_size=32000):
    """Train a SentencePiece BPE model, or load if it already exists"""
    model_path = f"{model_prefix}.model"
    if os.path.exists(model_path):
        print(f"Loading existing SentencePiece model: {model_path}")
    else:
        print(f"Training SentencePiece model: {model_path} (vocab_size={vocab_size})")
        # Write sentences to a temp file for SentencePiece training
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for sent in sentences:
                f.write(sent + '\n')
            tmp_path = f.name
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
        )
        os.remove(tmp_path)
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
    
def collate_fn(batch):
      en_batch, de_batch = zip(*batch)
      en_padded = pad_sequence(en_batch)
      de_padded = pad_sequence(de_batch)
      return torch.tensor(en_padded), torch.tensor(de_padded)




def load_wmt14(split="train", cache_dir="/home/jfan5/transformer/data"):
    """Load WMT14 de-en dataset; downloads from HuggingFace if not cached locally"""
    csv_path = os.path.join(cache_dir, f"wmt14_de-en_{split}.csv")

    if os.path.exists(csv_path):
        print(f"Loading cached {split} data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"Downloading WMT14 de-en {split} split...")
        from datasets import load_dataset
        ds = load_dataset("wmt14", "de-en", split=split)
        en_sentences = [item["translation"]["en"] for item in ds]
        de_sentences = [item["translation"]["de"] for item in ds]
        df = pd.DataFrame({"en": en_sentences, "de": de_sentences})
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    return df


def load_data(split="train", batch_size=32, tokenizer="bpe", vocab_size=32000):
    """
    Load dataset.
    tokenizer: "word" for word-level tokenization, "bpe" for SentencePiece BPE tokenization
    """
    df = load_wmt14(split=split)
    en_sentences = df["en"].tolist()
    de_sentences = df["de"].tolist()

    if tokenizer == "word":
        en_vocab = build_vocab(en_sentences)
        de_vocab = build_vocab(de_sentences)
        dataset = TranslationDataset(en_sentences, de_sentences, en_vocab, de_vocab)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        return dataloader, len(en_vocab), len(de_vocab)
    else:
        sp_dir = "/home/jfan5/transformer/data/spm"
        os.makedirs(sp_dir, exist_ok=True)
        en_sp = train_sentencepiece(en_sentences, f"{sp_dir}/en_bpe", vocab_size=vocab_size)
        de_sp = train_sentencepiece(de_sentences, f"{sp_dir}/de_bpe", vocab_size=vocab_size)
        dataset = BPETranslationDataset(en_sentences, de_sentences, en_sp, de_sp)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        return dataloader, en_sp.get_piece_size(), de_sp.get_piece_size()