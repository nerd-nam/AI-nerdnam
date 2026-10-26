import threading
import time
from types import SimpleNamespace

import pytest

from app import embedding


class FakeModel:
    """encode 가 동시에 몇 개 돌았는지 센다."""

    def __init__(self, device: str):
        self.device = SimpleNamespace(type=device)
        self.running = self.peak = 0
        self._count = threading.Lock()

    def encode(self, texts, normalize_embeddings):
        with self._count:
            self.running += 1
            self.peak = max(self.peak, self.running)
        time.sleep(0.05)
        with self._count:
            self.running -= 1
        return SimpleNamespace(tolist=lambda: [[0.0, 0.0] for _ in texts])


def run_concurrently(model, n=4):
    ts = [threading.Thread(target=embedding.encode, args=(model, "query", [("q", None)])) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return model.peak


@pytest.mark.parametrize("device,serialized", [("mps", True), ("cuda", False)])
def test_mps_only_serializes_encode(device, serialized):
    # 이슈 #22: MPS에서 동시 encode 는 프로세스를 죽이므로 하나씩만. CUDA는 동시 실행 유지
    peak = run_concurrently(FakeModel(device))
    assert (peak == 1) == serialized
