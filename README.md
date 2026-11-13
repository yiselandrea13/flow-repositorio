# mlflow-vinos

Clasificador de vinos (dataset `load_wine` de scikit-learn) con seguimiento de experimentos en MLflow.

## Instalar (Windows cmd)

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
pip freeze > requirements.txt
```

## Entrenar, registrar y predecir

```
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python src/train.py --n-estimators 50 --max-depth 3 --run-name probando
mlflow run . --env-manager local --experiment-name clasificacion-vinos -P n_estimators=200 -P max_depth=5
python src/register.py
python src/predict.py nuevos_vinos.csv --alias challenger
```

## Ver los resultados

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abre http://127.0.0.1:5000

## Estructura

- `src/config.py`: nombres del experimento y del modelo, semilla
- `src/data.py`: cargar y partir los datos
- `src/train.py`: entrena y registra en MLflow
- `src/register.py`: registra el mejor run como 'challenger'
- `src/predict.py`: carga el modelo por alias y predice
- `notebooks/01_exploracion.ipynb`: explorar y prototipar
- `data/`: datos pequeños de ejemplo
