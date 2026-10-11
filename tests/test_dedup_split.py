import pandas as pd

from inf8239_u02.data import dedup_key, deduplicate
from inf8239_u02.split import group_train_test_split


def test_dedup_key_normalizes_case_digits_and_spaces():
    assert dedup_key("WIN  £1000 now") == dedup_key("win £2500   NOW")


def test_deduplicate_removes_exact_copies():
    df = pd.DataFrame({"label": ["ham", "ham", "spam"], "text": ["hola", "hola", "gana"]})
    assert len(deduplicate(df)) == 2


def test_groups_never_cross_train_and_test():
    texts = [f"Call 0{i % 7} now to win" if i % 3 == 0 else f"see you at {i % 11} pm ok {i}"
             for i in range(200)]
    labels = ["spam" if i % 3 == 0 else "ham" for i in range(200)]
    df = deduplicate(pd.DataFrame({"label": labels, "text": texts}))
    train, test = group_train_test_split(df)
    assert set(train["group"]).isdisjoint(set(test["group"]))
