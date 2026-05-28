"""Token vocabulary shared by the synthetic corpus, the model and the scorer.

The corpus is word-level, so the vocabulary is a list of words. Index 0 is
reserved for the CTC blank, which means a word's token id is its position in
``words`` plus one. Keeping the blank at 0 is a convention the decoder, the
model output layer and the tests all rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

BLANK_TOKEN = "<blank>"
UNKNOWN_TOKEN = "<unk>"


class VocabularyError(ValueError):
    """Raised when a word is not in the vocabulary or an index is out of range."""


@dataclass(frozen=True)
class Vocabulary:
    """An ordered word list with the CTC blank at index 0."""

    words: tuple[str, ...]
    blank_index: int = 0
    _index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.words:
            raise VocabularyError("vocabulary must contain at least one word")
        if self.blank_index != 0:
            raise VocabularyError("blank must live at index 0")
        if len(set(self.words)) != len(self.words):
            raise VocabularyError("vocabulary contains duplicate words")
        mapping = {word: position + 1 for position, word in enumerate(self.words)}
        object.__setattr__(self, "_index", mapping)

    @property
    def size(self) -> int:
        """Number of classes the model predicts (words + blank)."""
        return len(self.words) + 1

    @property
    def num_words(self) -> int:
        """Number of real words, excluding the blank."""
        return len(self.words)

    def word_to_token(self, word: str) -> int:
        """Token id for ``word``."""
        try:
            return self._index[word]
        except KeyError as exc:
            raise VocabularyError(f"word {word!r} is not in the vocabulary") from exc

    def token_to_word(self, token: int) -> str:
        """Word for a token id (raises for the blank)."""
        if token == self.blank_index:
            raise VocabularyError("the blank token has no word")
        if not 1 <= token <= self.num_words:
            raise VocabularyError(f"token {token} is out of range")
        return self.words[token - 1]

    def contains(self, word: str) -> bool:
        """True when ``word`` is part of the vocabulary."""
        return word in self._index

    def encode(self, words: Iterable[str]) -> list[int]:
        """Encode a word sequence to token ids."""
        return [self.word_to_token(word) for word in words]

    def decode(self, tokens: Sequence[int]) -> list[str]:
        """Decode token ids to words, dropping blanks.

        Repeats are *not* collapsed here: collapsing is a decoding concern and
        lives in :mod:`asr.ctc`. This keeps ``decode`` usable for reference
        transcripts that legitimately repeat a word.
        """
        words: list[str] = []
        for token in tokens:
            if token == self.blank_index:
                continue
            words.append(self.token_to_word(token))
        return words

    def is_valid_token(self, token: int) -> bool:
        """True when ``token`` is the blank or a word id."""
        return token == self.blank_index or 1 <= token <= self.num_words

    def token_table(self) -> dict[int, str]:
        """Mapping of every token id to its display string."""
        table = {self.blank_index: BLANK_TOKEN}
        table.update({index + 1: word for index, word in enumerate(self.words)})
        return table

    def to_dict(self) -> dict[str, object]:
        """Serialisable form (stored next to model checkpoints)."""
        return {"words": list(self.words), "blank_index": self.blank_index}

    @classmethod
    def from_words(cls, words: Sequence[str]) -> "Vocabulary":
        """Build a vocabulary from a word list, preserving order."""
        return cls(words=tuple(str(word) for word in words))

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Vocabulary":
        """Rebuild a vocabulary from :meth:`to_dict` output."""
        words = payload.get("words")
        if not isinstance(words, list) or not words:
            raise VocabularyError("vocabulary payload must contain a non-empty 'words' list")
        return cls.from_words([str(word) for word in words])
