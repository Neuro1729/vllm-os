import pytest


@pytest.fixture(autouse=True)
def block_size_on_sequence():
    # Overrides tests/conftest.py, which sets Sequence.block_size = 4 for the CPU tests.
    # The GPU engine sets Sequence.block_size from the config (256), and a mismatch makes
    # block tables grow to tokens/4 entries. Leave the engine's value alone here.
    yield
