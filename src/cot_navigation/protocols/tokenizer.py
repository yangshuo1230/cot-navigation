"""Versioned vocabulary.

Node ids are whole tokens ``<V0>`` .. ``<V255>``. They are not split by a
natural-language BPE, and nothing is downloaded. Budget values used by any
shipped config are part of this version; adding one changes ``VOCAB_VERSION``.
"""

from __future__ import annotations

import hashlib

VOCAB_VERSION = "v1"
SUPPORTED_BUDGETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048)

_SPECIALS: tuple[str, ...] = (
    "<PAD>",
    "<GRAPH>",
    "<NODES>",
    "<EDGES>",
    "<E>",
    "<QUERY>",
    "<DIRECT>",
    "<PATH>",
    "<DFS>",
    "<FILLER>",
    "<BUDGET>",
    "<OUT>",
    "<ANS>",
    "<EOS>",
    "<CUT>",
    "<YES>",
    "<NO>",
    "<NONE>",
    "<STEP>",
    "<CUR>",
    "<CAND>",
    "<STACK>",
    "<VIS>",
    "<ACT>",
    "<ENDSTEP>",
    "<PUSH>",
    "<POP>",
    "<FOUND>",
    "<EXHAUST>",
    "<WALK>",
    "<DOT>",
)


class Tokenizer:
    def __init__(self, n_node_tokens: int = 256):
        names = list(_SPECIALS)
        names.extend(f"<T{budget}>" for budget in SUPPORTED_BUDGETS)
        self.node_lo = len(names)
        names.extend(f"<V{i}>" for i in range(n_node_tokens))
        self.itos = names
        self.stoi = {name: i for i, name in enumerate(names)}
        if len(self.stoi) != len(names):
            raise RuntimeError("duplicate token name")
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
        payload = (VOCAB_VERSION + "\n" + "\n".join(self.itos)).encode()
        return hashlib.sha256(payload).hexdigest()

    def id(self, name: str) -> int:
        try:
            return self.stoi[name]
        except KeyError as exc:
            raise KeyError(f"unknown token {name}") from exc

    def name(self, index: int) -> str:
        return self.itos[index]

    def budget_id(self, budget: int) -> int:
        if budget not in SUPPORTED_BUDGETS:
            raise KeyError(f"budget {budget} is not in vocab {VOCAB_VERSION}")
        return self.id(f"<T{budget}>")

    def node_id(self, raw: int) -> int:
        if not 0 <= raw < self.n_node_tokens:
            raise KeyError(f"node raw id {raw} out of range")
        return self.node_lo + raw

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def encode_tokens(self, names: list[str]) -> list[int]:
        return [self.id(name) for name in names]

    def mode_id(self, protocol: str) -> int:
        return self.id(
            {
                "direct": "<DIRECT>",
                "path": "<PATH>",
                "dfs_state": "<DFS>",
                "filler": "<FILLER>",
            }[protocol]
        )

    def label_id(self, label: int) -> int:
        if label not in (0, 1):
            raise ValueError(label)
        return self.yes if label == 1 else self.no
