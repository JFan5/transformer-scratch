import os
import tempfile
import pickle
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
    def __init__(self, en_sentences, de_sentences, en_sp, de_sp, cache_dir=None):
        cache_path = os.path.join(cache_dir, "bpe_tokens.pkl") if cache_dir else None

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached BPE tokens from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.en_tokens = cached["en_tokens"]
            self.de_tokens = cached["de_tokens"]
        else:
            print("Pre-encoding BPE tokens...")
            self.en_tokens = [
                [en_sp.bos_id()] + en_sp.encode(s) + [en_sp.eos_id()]
                for s in en_sentences
            ]
            self.de_tokens = [
                [de_sp.bos_id()] + de_sp.encode(s) + [de_sp.eos_id()]
                for s in de_sentences
            ]
            if cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({"en_tokens": self.en_tokens, "de_tokens": self.de_tokens}, f)
                print(f"Cached BPE tokens to {cache_path}")

    def __len__(self):
        return len(self.en_tokens)

    def __getitem__(self, idx):
        return self.en_tokens[idx], self.de_tokens[idx]


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
    
def collate_fn(batch, max_len=128):
      en_batch, de_batch = zip(*batch)
      # Truncate to max_len tokens
      en_batch = [seq[:max_len] for seq in en_batch]
      de_batch = [seq[:max_len] for seq in de_batch]
      en_padded = pad_sequence(en_batch)
      de_padded = pad_sequence(de_batch)
      return torch.tensor(en_padded), torch.tensor(de_padded)




def load_wmt14(split="train", cache_dir=os.path.expanduser("~/transformer/data")):
    """Load WMT14 de-en dataset; downloads from HuggingFace if not cached locally"""
    parquet_path = os.path.join(cache_dir, f"wmt14_de-en_{split}.parquet")

    if os.path.exists(parquet_path):
        print(f"Loading cached {split} data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"Downloading WMT14 de-en {split} split...")
        from datasets import load_dataset
        ds = load_dataset("wmt14", "de-en", split=split)
        en_sentences = [item["translation"]["en"] for item in ds]
        de_sentences = [item["translation"]["de"] for item in ds]
        df = pd.DataFrame({"en": en_sentences, "de": de_sentences})
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        print(f"Saved to {parquet_path}")

    return df


def load_data(split="train", batch_size=32, tokenizer="bpe", vocab_size=32000, max_len=128, max_samples=None):
    """
    Load dataset.
    tokenizer: "word" for word-level tokenization, "bpe" for SentencePiece BPE tokenization
    max_samples: if set, randomly sample this many sentence pairs
    """
    df = load_wmt14(split=split)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {max_samples} sentence pairs from {split} set")
    en_sentences = df["en"].tolist()
    de_sentences = df["de"].tolist()

    if tokenizer == "word":
        en_vocab = build_vocab(en_sentences)
        de_vocab = build_vocab(de_sentences)
        dataset = TranslationDataset(en_sentences, de_sentences, en_vocab, de_vocab)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, max_len=max_len), shuffle=True)
        return dataloader, len(en_vocab), len(de_vocab)
    else:
        sp_dir = os.path.expanduser("~/transformer/data/spm")
        os.makedirs(sp_dir, exist_ok=True)
        en_sp = train_sentencepiece(en_sentences, f"{sp_dir}/en_bpe", vocab_size=vocab_size)
        de_sp = train_sentencepiece(de_sentences, f"{sp_dir}/de_bpe", vocab_size=vocab_size)
        cache_dir = os.path.expanduser("~/transformer/data/spm")
        dataset = BPETranslationDataset(en_sentences, de_sentences, en_sp, de_sp, cache_dir=cache_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_len),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return dataloader, en_sp.get_piece_size(), de_sp.get_piece_size()