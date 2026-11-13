"""Carga y partición de datos. Si cambian los datos, se cambia solo este archivo."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from config import SEMILLA


def cargar_datos(test_size=0.3):
    """Devuelve X_train, X_test, y_train, y_test con la misma partición siempre."""
    X, y = load_wine(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=test_size, random_state=SEMILLA, stratify=y)
