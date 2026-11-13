"""Demo de las partes C y D de la guía: guardar y consultar información en MLflow."""
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("demo-referencia")

X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
modelo = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)
firma = infer_signature(X_train, modelo.predict(X_train))
estado = {}


def parametros():
    with mlflow.start_run(run_name="demo-parametros"):
        mlflow.log_param("modelo", "RandomForest")
        mlflow.log_params({"n_estimators": 100, "max_depth": 4, "seed": 42})


def metricas():
    pred = modelo.predict(X_test)
    with mlflow.start_run(run_name="demo-metricas"):
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        mlflow.log_metrics({
            "f1_macro": f1_score(y_test, pred, average="macro"),
            "n_filas_prueba": len(y_test),
        })
        for epoca in range(5):
            mlflow.log_metric("perdida", 1.0 / (epoca + 1), step=epoca)


def etiquetas():
    with mlflow.start_run(run_name="demo-etiquetas"):
        mlflow.set_tag("autor", "yisel")
        mlflow.set_tags({"dataset": "wine", "etapa": "desarrollo"})
        mlflow.set_tag("mlflow.note.content", "Primer intento con 100 arboles")


def artefactos():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with mlflow.start_run(run_name="demo-artefactos"):
            muestra = os.path.join(tmp, "muestra.csv")
            X_test.head(20).to_csv(muestra, index=False)
            mlflow.log_artifact(muestra)
            mlflow.log_artifact(muestra, artifact_path="datos")

            carpeta = os.path.join(tmp, "reportes")
            os.makedirs(carpeta, exist_ok=True)
            with open(os.path.join(carpeta, "resumen.txt"), "w", encoding="utf-8") as f:
                f.write("accuracy alto, sin sobreajuste evidente")
            mlflow.log_artifacts(carpeta, artifact_path="reportes")

            mlflow.log_text("Notas de la ejecucion", "notas.txt")
            mlflow.log_dict({"clases": ["clase_0", "clase_1", "clase_2"], "seed": 42}, "config.json")

            fig, ax = plt.subplots()
            ax.bar(X_train.columns[:5], modelo.feature_importances_[:5])
            ax.set_title("Importancia de variables (primeras 5)")
            ax.tick_params(axis="x", rotation=45)
            mlflow.log_figure(fig, "importancias.png")
            plt.close(fig)

            mlflow.log_table(
                data=X_test.head(10).assign(real=y_test.head(10).values),
                artifact_file="tabla_prueba.json",
            )
            mlflow.log_image(np.zeros((32, 32, 3), dtype=np.uint8), "imagen.png")


def dataset():
    df_train = X_train.assign(target=y_train)
    ds = mlflow.data.from_pandas(
        df_train, source="sklearn.datasets.load_wine", name="wine-train", targets="target"
    )
    with mlflow.start_run(run_name="demo-dataset"):
        mlflow.log_input(ds, context="training")


def guardar_modelo():
    with mlflow.start_run(run_name="demo-modelo") as run:
        info = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=firma,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.set_tag("model_uri", info.model_uri)
        estado["uri"] = info.model_uri
        print("URI del modelo:", info.model_uri, "| Run ID:", run.info.run_id)


def autolog():
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True, silent=True)
    try:
        with mlflow.start_run(run_name="demo-autolog"):
            RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42).fit(X_train, y_train)
    finally:
        mlflow.sklearn.autolog(disable=True)


def anidados():
    with mlflow.start_run(run_name="busqueda-profundidad"):
        for profundidad in [2, 4, 6]:
            with mlflow.start_run(run_name=f"depth={profundidad}", nested=True):
                m = RandomForestClassifier(max_depth=profundidad, random_state=42).fit(X_train, y_train)
                mlflow.log_param("max_depth", profundidad)
                mlflow.log_metric("accuracy", m.score(X_test, y_test))


class ModeloConUmbral(mlflow.pyfunc.PythonModel):
    """Envuelve un modelo para devolver tambien la confianza."""

    def __init__(self, modelo, umbral=0.6):
        self.modelo = modelo
        self.umbral = umbral

    def predict(self, context, model_input, params=None):
        proba = self.modelo.predict_proba(model_input)
        resultado = pd.DataFrame({
            "clase": proba.argmax(axis=1),
            "confianza": proba.max(axis=1),
        })
        resultado["revisar"] = resultado["confianza"] < self.umbral
        return resultado


def pyfunc_propio():
    with mlflow.start_run(run_name="demo-pyfunc"):
        mlflow.pyfunc.log_model(
            name="modelo_con_umbral",
            python_model=ModeloConUmbral(modelo, umbral=0.6),
            input_example=X_train.head(3),
            pip_requirements=["scikit-learn", "pandas"],
        )


def evaluar():
    datos_prueba = X_test.assign(target=y_test)
    with mlflow.start_run(run_name="demo-evaluacion"):
        resultado = mlflow.models.evaluate(
            model=estado["uri"], data=datos_prueba, targets="target", model_type="classifier"
        )
        print("Metricas de evaluacion:", resultado.metrics)


def guardar_carpeta():
    if os.path.exists("modelo_local"):
        shutil.rmtree("modelo_local")
    mlflow.sklearn.save_model(
        modelo,
        path="modelo_local",
        signature=firma,
        input_example=X_train.head(3),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    cargado = mlflow.pyfunc.load_model("modelo_local")
    print("Prediccion desde carpeta:", cargado.predict(X_test.head(3)))


def buscar():
    runs = mlflow.search_runs(
        experiment_names=["demo-referencia"],
        filter_string="metrics.accuracy > 0.9 AND params.max_depth = '4'",
        order_by=["metrics.accuracy DESC"],
        max_results=10,
    )
    print(runs[["run_id", "tags.mlflow.runName", "params.max_depth", "metrics.accuracy"]])

    cliente = MlflowClient()
    run_id = mlflow.search_runs(
        experiment_names=["demo-referencia"],
        filter_string="tags.mlflow.runName = 'demo-metricas'",
    ).iloc[0]["run_id"]
    run = cliente.get_run(run_id)
    print(run.data.params, run.data.metrics, run.info.status)
    for punto in cliente.get_metric_history(run_id, "perdida"):
        print("step", punto.step, round(punto.value, 3))
    for archivo in cliente.list_artifacts(run_id):
        print("artefacto:", archivo.path)


def descargar():
    run_id = mlflow.search_runs(
        experiment_names=["demo-referencia"],
        filter_string="tags.mlflow.runName = 'demo-artefactos'",
    ).iloc[0]["run_id"]
    ruta = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="config.json", dst_path="descargas"
    )
    print("Descargado en:", ruta)


def cargar():
    fila = mlflow.search_runs(
        experiment_names=["demo-referencia"],
        filter_string="tags.mlflow.runName = 'demo-modelo'",
    ).iloc[0]
    uri = fila["tags.model_uri"]
    m_pyfunc = mlflow.pyfunc.load_model(uri)
    print(m_pyfunc.predict(X_test.head(3)))
    m_sk = mlflow.sklearn.load_model(uri)
    print(m_sk.predict_proba(X_test.head(1)))
    print(m_pyfunc.metadata.get_input_schema())


if __name__ == "__main__":
    pasos = [parametros, metricas, etiquetas, artefactos, dataset, guardar_modelo, autolog,
             anidados, pyfunc_propio, evaluar, guardar_carpeta, buscar, descargar, cargar]
    for paso in pasos:
        print(f"\n=== {paso.__name__} ===")
        try:
            paso()
            print("OK")
        except Exception as e:
            print(f"FALLO: {type(e).__name__}: {str(e)[:300]}")