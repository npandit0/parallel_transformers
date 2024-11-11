from typing import List
# sentencepeice is a library
from sentencepiece import SentencePieceProcessor
import pdb


class MistralTokenizer:
    def __init__(self, model_path: str):
        # pdb.set_trace()
        self._model = SentencePieceProcessor(model_file=str(model_path)) # cahnge to string, versioning problem?

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)
