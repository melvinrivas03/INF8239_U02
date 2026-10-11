from pathlib import Path

import pytest

from inf8239_u02.data import DEFAULT_PATH, LABELS, load_corpus

pytestmark = pytest.mark.skipif(not Path(DEFAULT_PATH).exists(),
                                reason="Ejecutar primero la descarga del corpus")


def test_corpus_has_expected_columns_and_size():
    df = load_corpus()
    assert list(df.columns) == ["label", "text"]
    assert len(df) >= 5000


def test_labels_are_ham_or_spam():
    assert set(load_corpus()["label"].unique()) == set(LABELS)


def test_no_empty_messages():
    assert (load_corpus()["text"].str.strip() != "").all()
