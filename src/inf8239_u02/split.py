"""Partición entrenamiento/prueba que impide que un mismo grupo de casi duplicados cruce conjuntos."""
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42


def group_train_test_split(frame, n_splits: int = 5, random_state: int = RANDOM_STATE):
    """Devuelve (train, test) ≈ 80/20, estratificado por etiqueta y sin compartir grupos."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(frame["text"], frame["label"], frame["group"]))
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
