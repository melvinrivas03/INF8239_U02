# INF-8239 · Unidad 02 — Ejercicio 03: corpus público y clasificador de texto

**Estudiante:** Melvin Humberto Rivas Genao

## Problema
Clasificar SMS como **spam** o **ham** sin bloquear mensajes legítimos.
- **Corpus:** SMS Spam Collection — UCI #228, CC BY 4.0 (`docs/ficha_corpus.md`).
- **Prevención de fuga:** eliminación de 403 duplicados exactos y partición 80/20 por grupos de casi duplicados (`StratifiedGroupKFold`), TF-IDF dentro del pipeline.
- **Modelos:** baseline (siempre ham), TF-IDF + Naive Bayes y TF-IDF + regresión logística, con la misma vectorización.
- **Modelo seleccionado:** TF-IDF + Naive Bayes (0 falsos positivos en prueba).
- **Uso de IA:** `docs/declaracion_ia.md`.

## Estructura
```
data/raw/         corpus descargado (no se versiona)
docs/             ficha del corpus y declaración de uso de IA
notebooks/        03_corpus_clasificador_texto.ipynb
reports/          tablas CSV, figuras y registro del entorno
src/inf8239_u02/  data.py · split.py · models.py
tests/            pruebas con pytest
```

## Reproducción
1. Abrir `notebooks/03_corpus_clasificador_texto.ipynb` en Google Colab.
2. Definir el secreto `GITHUB_TOKEN` (permiso *Contents: Read and write* sobre este repositorio).
3. Ejecutar todas las celdas en orden. La descarga del corpus es automática:
```python
from inf8239_u02.data import download_corpus
download_corpus()   # UCI -> data/raw/sms_spam.csv
```
Ejecución local equivalente:
```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -c "from inf8239_u02.data import download_corpus; download_corpus()"
PYTHONPATH=src python -m pytest -q
```

## Resultados principales (prueba, 1,031 mensajes)
| Modelo | F1 spam | Precisión spam | Recall spam | FP | FN |
|---|---|---|---|---|---|
| Baseline | 0.000 | — | 0.000 | 0 | 134 |
| TF-IDF + Naive Bayes | 0.932 | 1.000 | 0.873 | 0 | 17 |
| TF-IDF + Regresión logística | 0.951 | 0.962 | 0.940 | 5 | 8 |
