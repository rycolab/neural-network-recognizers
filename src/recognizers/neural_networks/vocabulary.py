import dataclasses
import pathlib
from typing import Optional

import torch

from rau.tasks.language_modeling.vocabulary import build_embedding_vocab
from rau.vocab import Vocabulary, VocabularyBuilder, ToStringVocabularyBuilder

@dataclasses.dataclass
class VocabularyData:
    tokens: list[str]
    allow_unk: bool

def load_vocabulary_data_from_file(path: pathlib.Path) -> VocabularyData:
    data = torch.load(path)
    return VocabularyData(data['tokens'], data['allow_unk'])

def get_vocabularies(
    vocabulary_data: VocabularyData,
    use_bos: bool,
    use_eos: bool,
    builder: Optional[VocabularyBuilder]=None
) -> tuple[Vocabulary, Vocabulary]:
    if builder is None:
        builder = ToStringVocabularyBuilder()
    softmax_vocab = build_softmax_vocab(
        vocabulary_data.tokens,
        vocabulary_data.allow_unk,
        use_eos,
        builder
    )
    embedding_vocab = build_embedding_vocab(
        softmax_vocab,
        use_bos,
        builder
    )
    return embedding_vocab, softmax_vocab

def build_softmax_vocab(tokens, allow_unk, use_eos, builder):
    result = builder.content(tokens)
    if allow_unk:
        result = result + builder.catchall('unk')
    if use_eos:
        result = result + builder.reserved(['eos'])
    return result
