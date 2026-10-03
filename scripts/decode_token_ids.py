import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.tokenizer import Tokenizer
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

tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# Build a lookup from special token string -> token ID
special_token_ids = {
    t: tokenizer.byte_to_id[t.encode("utf-8")]
    for t in tokenizer.special_tokens
}

print("Enter token IDs")
for t, tid in special_token_ids.items():
    print(f'  Special token: "{t}" -> ID {tid}')
print()

while True:
    try:
        raw = input("Token IDs: ")
    except (EOFError, KeyboardInterrupt):
        print()
        break

    raw = raw.strip()
    if not raw:
        continue

    parts = raw.split()
    ids = []
    for p in parts:
        if p in special_token_ids:
            ids.append(special_token_ids[p])
        else:
            try:
                ids.append(int(p))
            except ValueError:
                print(f'  Skipping invalid input: "{p}"')
                continue

    # Validate all IDs are in vocab
    invalid = [i for i in ids if i not in tokenizer.vocab]
    if invalid:
        print(f"  Invalid token ID（ Not in vocab）: {invalid}")
        print(f"  Vocab Range: 0 ~ {max(tokenizer.vocab.keys())}")
        continue

    tokens = [tokenizer.decode([i]) for i in ids]
    text = tokenizer.decode(ids)

    print(f"  Tokens:  {tokens}")
    print(f"  Decoded: {repr(text)}")
    print(f"  Count:   {len(ids)}")
    print()
