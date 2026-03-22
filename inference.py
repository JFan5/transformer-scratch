import os
import torch
import sentencepiece as spm
from model import Transformer
from masks import create_padding_mask, create_causal_mask


def translate(model, text, en_sp, de_sp, device, max_len=128):
    """Greedy decoding: generate translation one token at a time"""
    model.eval()

    # Encode source: add BOS/EOS
    src_ids = [en_sp.bos_id()] + en_sp.encode(text) + [en_sp.eos_id()]
    src_tensor = torch.tensor([src_ids], device=device)
    src_mask = create_padding_mask(src_tensor)

    # Encode source once
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)

    # Start with BOS token
    tgt_ids = [de_sp.bos_id()]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids], device=device)
        tgt_mask = create_padding_mask(tgt_tensor) & create_causal_mask(tgt_tensor)

        with torch.no_grad():
            dec_output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            logits = model.output_linear(dec_output)

        # Take the last token's prediction
        next_token = logits[:, -1, :].argmax(dim=-1).item()

        if next_token == de_sp.eos_id():
            break

        tgt_ids.append(next_token)

    # Decode token IDs back to text (skip BOS)
    translation = de_sp.decode(tgt_ids[1:])
    return translation


if __name__ == "__main__":
    # Config must match training
    config = {
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
        "num_layers": 3,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SentencePiece models
    sp_dir = os.path.expanduser("~/transformer/data/spm")
    en_sp = spm.SentencePieceProcessor()
    en_sp.load(f"{sp_dir}/en_bpe.model")
    de_sp = spm.SentencePieceProcessor()
    de_sp.load(f"{sp_dir}/de_bpe.model")

    # Load model
    model = Transformer(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        src_vocab_size=en_sp.get_piece_size(),
        tgt_vocab_size=de_sp.get_piece_size(),
    )
    model.load_state_dict(torch.load("best_model.pt", weights_only=True, map_location=device))
    model.to(device)

    # Interactive translation
    print("English -> German Translator (type 'quit' to exit)")
    while True:
        text = input("\nEnglish: ").strip()
        if text.lower() == "quit":
            break
        result = translate(model, text, en_sp, de_sp, device)
        print(f"German:  {result}")
