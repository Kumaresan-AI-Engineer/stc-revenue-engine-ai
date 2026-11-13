import threading

from core.config import Settings
from llm.local_embedder import LocalEmbedder


class StubEngine:
    def __init__(self) -> None:
        self.threads: list[int] = []

    def embed(self, texts: list[str], batch_size: int) -> list[list[float]]:
        self.threads.append(threading.get_ident())
        return [[float(len(t)), 1.0] for t in texts]


async def test_embeds_in_order_off_the_event_loop() -> None:
    embedder = LocalEmbedder(Settings())
    engine = StubEngine()
    embedder.__dict__["_engine"] = engine  # replace the ONNX model
    assert await embedder.embed(["a", "abc"]) == [[1.0, 1.0], [3.0, 1.0]]
    assert engine.threads and engine.threads[0] != threading.get_ident()
    assert embedder.model == "BAAI/bge-small-en-v1.5"


async def test_no_texts_means_no_model_call() -> None:
    embedder = LocalEmbedder(Settings())
    engine = StubEngine()
    embedder.__dict__["_engine"] = engine
    assert await embedder.embed([]) == []
    assert engine.threads == []
