"""Compact symbolic vocabulary v2, incompatible with old checkpoints.

Each node is one token. Roles come from EDGE/QUERY and fixed DFS field order;
no graph geometry is encoded in embeddings.
"""
from __future__ import annotations
import hashlib

VOCAB_VERSION = "v2"
_SPECIALS = (
    "<PAD>", "<EOS>", "<EDGE>", "<QUERY>", "|", ";",
    "<YES>", "<NO>", "<PUSH>", "<POP>", "<FOUND>", "<EXHAUST>",
    "<WALK>", "<DOT>",
)

class Tokenizer:
    def __init__(self, n_node_tokens: int = 256):
        if n_node_tokens < 1:
            raise ValueError("n_node_tokens must be positive")
        self.node_lo = len(_SPECIALS)
        self.itos = list(_SPECIALS) + [f"<V{i}>" for i in range(n_node_tokens)]
        self.stoi = {name: i for i, name in enumerate(self.itos)}
        self.pad = self.stoi["<PAD>"]
        self.yes = self.stoi["<YES>"]
        self.no = self.stoi["<NO>"]
        self.n_node_tokens = n_node_tokens

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def version(self) -> str:
        return VOCAB_VERSION

    def hash(self) -> str:
        return hashlib.sha256((VOCAB_VERSION + "\n" + "\n".join(self.itos)).encode()).hexdigest()

    def id(self, name: str) -> int:
        try:
            return self.stoi[name]
        except KeyError as exc:
            raise KeyError(f"unknown token {name}") from exc

    def name(self, index: int) -> str:
        return self.itos[index]

    def node_id(self, raw: int) -> int:
        if not 0 <= raw < self.n_node_tokens:
            raise KeyError(f"node raw id {raw} out of range")
        return self.node_lo + raw

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def encode_tokens(self, names: list[str]) -> list[int]:
        return [self.id(name) for name in names]

    def label_id(self, label: int) -> int:
        if label not in (0, 1):
            raise ValueError(label)
        return self.yes if label == 1 else self.no
