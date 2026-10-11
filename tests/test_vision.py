from pathlib import Path

import numpy as np
import pytest
import torch

from inf8239_u02.vision import (CLASSES, DATA_DIR, SmallCNN, count_params, load_fashion_mnist,
                                split_train_val)


def test_cnn_outputs_ten_logits_per_image():
    out = SmallCNN()(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, len(CLASSES))


def test_cnn_is_small():
    assert count_params(SmallCNN()) < 100_000


def test_validation_split_is_stratified_and_disjoint():
    x = np.arange(1000).reshape(1000, 1, 1)
    y = np.repeat(np.arange(10), 100)
    xtr, xva, ytr, yva = split_train_val(x, y)
    assert len(xva) == 100 and set(np.bincount(yva)) == {10}
    assert not set(xtr.ravel()) & set(xva.ravel())


@pytest.mark.skipif(not Path(DATA_DIR).exists(), reason="Ejecutar primero la descarga")
def test_data_contract():
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    assert x_train.shape == (60000, 28, 28) and x_test.shape == (10000, 28, 28)
    assert set(np.unique(y_train)) == set(range(10))
