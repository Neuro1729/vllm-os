import pytest

from nanovllm.engine.sequence import Sequence
from helpers import BLOCK_SIZE


@pytest.fixture(autouse=True)
def block_size_on_sequence():
    # LLMEngine sets Sequence.block_size from the config; do the same here.
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous
