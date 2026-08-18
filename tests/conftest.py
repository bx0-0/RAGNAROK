import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: marks tests that require a running gateway (deselect with '-m \"not live\"')",
    )
