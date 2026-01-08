import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timing() -> Iterator[float]:
    start = time.perf_counter()
    try:
        yield start
    finally:
        _ = time.perf_counter() - start
