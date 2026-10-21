"""The small CNN baseline: shape, determinism and prediction (tiny random data)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mri_pipeline import cnn  # noqa: E402


@pytest.fixture
def tiny() -> tuple[object, np.ndarray]:
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, size=(16, 16), dtype=np.uint8) for _ in range(10)]
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    return cnn.to_uint8_tensor(images), labels


def test_model_shape_and_parameter_count():
    model = cnn.build_model(4)
    out = model(torch.zeros(2, 1, 32, 32))
    assert tuple(out.shape) == (2, 4)
    assert cnn.count_parameters(model) == sum(p.numel() for p in model.parameters()) > 0


def test_to_uint8_tensor_layout(tiny):
    x, _ = tiny
    assert tuple(x.shape) == (10, 1, 16, 16)
    assert x.dtype == torch.uint8


def test_training_is_reproducible_with_the_same_seed(tiny):
    x, y = tiny
    logged = []
    _, first = cnn.train(x, y, n_classes=4, epochs=2, seed=5, batch_size=4, log=logged.append)
    _, second = cnn.train(x, y, n_classes=4, epochs=2, seed=5, batch_size=4)
    strip = [{k: v for k, v in h.items() if k != "seconds"} for h in first]
    assert strip == [{k: v for k, v in h.items() if k != "seconds"} for h in second]
    assert [h["epoch"] for h in first] == [1, 2]
    assert logged == first
    assert all(0.0 <= h["train_accuracy_running"] <= 1.0 for h in first)


def test_predict_returns_one_class_per_image(tiny):
    x, y = tiny
    model, _ = cnn.train(x, y, n_classes=4, epochs=1, seed=0, batch_size=8, flip=False)
    pred = cnn.predict(model, x, batch_size=3)
    assert pred.shape == (10,)
    assert set(pred.tolist()) <= {0, 1, 2, 3}


def test_same_seed_gives_identical_weights(tiny):
    x, y = tiny
    first, _ = cnn.train(x, y, n_classes=4, epochs=2, seed=3, batch_size=4, threads=2)
    second, _ = cnn.train(x, y, n_classes=4, epochs=2, seed=3, batch_size=4, threads=2)
    a, b = first.state_dict(), second.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 2


def test_thread_count_from_environment():
    assert cnn.thread_count({}) == 4
    assert cnn.thread_count({cnn.THREADS_ENV: "6"}) == 6
    with pytest.raises(ValueError):
        cnn.thread_count({cnn.THREADS_ENV: "0"})


def test_a_last_batch_of_one_image_is_skipped():
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 256, size=(8, 8), dtype=np.uint8) for _ in range(9)]
    x, y = cnn.to_uint8_tensor(images), np.arange(9) % 4
    _, history = cnn.train(x, y, n_classes=4, epochs=1, seed=0, batch_size=4)  # 4 + 4 + 1
    assert len(history) == 1 and history[0]["train_loss"] > 0
    with pytest.raises(ValueError):
        cnn.train(x[:1], y[:1], n_classes=4, epochs=1, seed=0)
