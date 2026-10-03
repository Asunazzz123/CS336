import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.tokenizer import TokenizerNoRegex, TokenizerWithRegex
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode


gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

with open(FIXTURES_PATH / "gpt2_vocab.json") as f:
    gpt2_vocab = json.load(f)

vocab = {
    idx: bytes([gpt2_byte_decoder[t] for t in token])
    for token, idx in gpt2_vocab.items()
}

merges = []
with open(FIXTURES_PATH / "gpt2_merges.txt") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        if len(parts) == 2:
            merges.append((
                bytes([gpt2_byte_decoder[t] for t in parts[0]]),
                bytes([gpt2_byte_decoder[t] for t in parts[1]]),
            ))

tokenizers = {
    "with regex": TokenizerWithRegex(vocab, merges, special_tokens=["<|endoftext|>"]),
    "no regex": TokenizerNoRegex(vocab, merges, special_tokens=["<|endoftext|>"]),
}


while True:
    try:
        text = input("\n输入文本 (Ctrl+C 退出): ")
    except (EOFError, KeyboardInterrupt):
        break

    for name, tokenizer in tokenizers.items():
        ids = tokenizer.encode(text)
        tokens = [tokenizer.decode([i]) for i in ids]

        print(f"\n[{name}]")
        print(f"Token IDs:  {ids}")
        print(f"Tokens:     {tokens}")
        print(f"Decoded:    {tokenizer.decode(ids)}")
        print(f"Token 数量: {len(ids)}")
