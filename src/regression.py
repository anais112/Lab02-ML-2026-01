from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(slots=True)
class AgeRegressionGuide:
    """Resume los pasos que deben seguir los estudiantes para la regresion."""

    scoring: str = "neg_mean_absolute_error"
    suggested_metrics: tuple[str, ...] = ("MAE", "RMSE", "R2")

    def to_text(self) -> str:
        """Genera un recordatorio corto para acompanar el laboratorio."""

        return (
            "Guia de regresion de edad\n"
            "========================\n"
            "\n"
            "La parte de regresion no se implementa en esta version.\n"
            "Los estudiantes deben completar src/regression.py siguiendo estos pasos:\n"
            "\n"
            "1. Reutilizar la misma matriz X preprocesada.\n"
            "2. Construir un Pipeline con PCA + LinearRegression.\n"
            "3. Ajustar pca__n_components con GridSearchCV.\n"
            "4. Evaluar con MAE, RMSE y R2.\n"
            "5. Guardar el modelo cuando este listo.\n"
        )


def build_age_regression_pipeline(random_state: int) -> Any:
    pipe = Pipeline([
        ("pca", PCA(whiten=True, random_state=random_state)),
        ("reg", LinearRegression()),
    ])
    return pipe


def train_age_regressor(
    X_train: Any,
    y_age_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
) -> Any:
    pipe = build_age_regression_pipeline(random_state)

    param_grid = {
        "pca__n_components": pca_components
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_age_train)

    return grid.best_estimator_


def evaluate_age_regressor(model: Any, X_test: Any, y_age_test: Any) -> dict[str, float]:
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_age_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_age_test, y_pred))
    r2   = r2_score(y_age_test, y_pred)

    return {
        "MAE":  round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2":   round(r2, 4)
    }


def save_age_regressor(model: Any, output_path: str) -> None:
    joblib.dump(model, output_path)
    print(f"Modelo de edad guardado en {output_path}")