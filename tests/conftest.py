"""Tests must not depend on which model weights happen to be cached on the machine."""
import pytest
from backend.learned import registry


@pytest.fixture(autouse=True)
def _learned_models_off():
    registry.set_mode("off")
    registry.reset()
    yield
    registry.set_mode(None)
    registry.reset()
