import re
import os
import regex
from collections import OrderedDict


GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class _BaseTokenizer:
    def __init__(self,vocab,merges,special_tokens = None):
        """Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token."""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks_bytes = {pair[0]+pair[1]: rank for rank, pair in enumerate(merges)}
        self._special_token_to_id = {
            token: self.byte_to_id[token.encode("utf-8")]
            for token in self.special_tokens
            if token.encode("utf-8") in self.byte_to_id
        }

    def _token_split_special_token(self,text: str) -> list[tuple[str,bool]]:
        if not self.special_tokens:
            return [(text, False)]
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")"
        parts = re.split(pattern, text)
        special_set = set(sorted_special_tokens)
        return [
            (part, part in special_set)
            for part in parts
            if part != ""
        ]

    def _bpe_encoder(self,raw_byte: bytes):
        tokens = [bytes([b]) for b in raw_byte]

        while True:
            if len(tokens) < 2:
                break
            pairs = [tokens[i] + tokens[i+1] for i in range(len(tokens) - 1)]
            candidate = [pair for pair in pairs if pair in self.merge_ranks_bytes]

            if not candidate:
                break

            best_pair = min(candidate, key = lambda pair: self.merge_ranks_bytes[pair])

            new_tokens = []
            i = 0
            while i < len(tokens):
                if ( i < len(tokens) - 1 and tokens[i] + tokens[i+1] == best_pair):
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def _encode_piece(self, piece: str) -> list[int]:
        token_bytes = piece.encode("utf-8")
        tokens = self._bpe_encoder(token_bytes)
        return [self.byte_to_id[token] for token in tokens]

    def encode(self,text: str) -> list[int]:
        token_ids = []
        for piece, is_special in self._token_split_special_token(text):
            if piece == "":
                continue

            if is_special:
                token_ids.append(self._special_token_to_id[piece])

            else:
                token_ids.extend(self._encode_piece(piece))
        return token_ids


        # pass

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self,ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return raw_bytes.decode("utf-8", errors="replace")
        # pass


class TokenizerNoRegex(_BaseTokenizer):
    """BPE tokenizer that applies BPE directly to each non-special text span."""


class TokenizerWithRegex(_BaseTokenizer):
    """GPT-2-style BPE tokenizer with regex pre-tokenization."""

    def _encode_piece(self, piece: str) -> list[int]:
        token_ids = []
        for match in GPT2_PRETOKEN_PATTERN.finditer(piece):
            token_bytes = match.group().encode("utf-8")
            tokens = self._bpe_encoder(token_bytes)
            token_ids.extend(self.byte_to_id[token] for token in tokens)
        return token_ids


Tokenizer = TokenizerWithRegex



def train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    use_pretokenization: bool = True,
    **kwargs,
):
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    vocab = {i: bytes([i]) for i in range(256)}
    

    if special_tokens:
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")
        special_pattern = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        chunks = regex.split(special_pattern, text)
    else:
        chunks = [text]

    word_counts = {}
    for chunk in chunks:
        if use_pretokenization:
            pieces = (match.group() for match in GPT2_PRETOKEN_PATTERN.finditer(chunk))
        else:
            pieces = (chunk,) if chunk else ()
        for piece in pieces:
            token = tuple(bytes([b]) for b in piece.encode("utf-8"))
            word_counts[token] = word_counts.get(token, 0) + 1

    merges = []
    while len(vocab) < vocab_size:
        pair_counts = {}
        for word, count in word_counts.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count

        if not pair_counts:
            break
        
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = new_token

        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = None
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    if new_word is None:
                        new_word = list(word[:i])
                    new_word.append(new_token)
                    i += 2
                else:
                    if new_word is not None:
                        new_word.append(word[i])
                    i += 1
            if new_word is None:
                new_word_counts[word] = new_word_counts.get(word, 0) + count
            else:
                new_word = tuple(new_word)
                new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count
        word_counts = new_word_counts

    return vocab, merges
