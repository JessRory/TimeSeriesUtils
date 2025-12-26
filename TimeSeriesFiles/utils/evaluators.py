import numpy as np
import pandas as pd # type: ignore
from darts import TimeSeries, concatenate # type: ignore
from darts.metrics import smape # type: ignore
from sklearn.model_selection import ParameterGrid # type: ignore
from typing import Optional, Any, Tuple # type: ignore
from ..utilities import DartsPipeline # type: ignore

class TimeSeriesEvaluator:
    
    @staticmethod
    def walk_forward_backtest(
        pipeline: DartsPipeline, # type: ignore
        series: TimeSeries,
        forecast_horizon: int,
        train_length: Optional[int] = None,
        step: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[float, TimeSeries]:
        
        """
        Realiza un backtest de avance progresivo (walk-forward) en la serie temporal dada.
        Parámetros:
        - pipeline: DartsPipeline, Pipeline que contiene el modelo y el transformador.
        - series: TimeSeries, serie temporal completa para el backtest.
        - forecast_horizon: int, número de pasos a predecir en cada iteración
        - train_length: Optional[int], longitud inicial del conjunto de entrenamiento. Si es None, se usa forecast_horizon * 2.
        - step: Optional[int], número de pasos para avanzar el conjunto de entrenamiento en cada iteración. Si es None, se usa forecast_horizon.
        - verbose: bool, si es True, imprime el SMAPE en cada iteración.
        
        Devuelve:
        - Tuple[float, TimeSeries]: SMAPE promedio y las predicciones concatenadas.
        """
        # Creamos listas para almacenar errores y predicciones:
        errors = []
        predictions = []
        total_length = len(series)
        
        # Condiciones iniciales:
        if train_length is None:
            train_length = forecast_horizon * 2
        step = step or forecast_horizon
        start = train_length

        # Bucle de avance progresivo:
        # Mientras haya suficiente serie para entrenar y predecir:
        while start + forecast_horizon <= total_length:
            # Dividimos la serie en entrenamiento y validación:
            train = series[:start]
            val = series[start:start + forecast_horizon]

            # y Re-instanciamos el modelo para cada iteración (limpieza)
            model_copy = pipeline.model.__class__(**pipeline.model._model_params) # type: ignore
            transformer_copy = pipeline.transformer.__class__() if pipeline.transformer else None # type: ignore

            # Instanciamos una nueva Pipeline para cada iteración y ajustamos:
            new_pipeline = DartsPipeline(model_copy, transformer_copy) # type: ignore
            new_pipeline.fit(train) # type: ignore
            pred = new_pipeline.predict(forecast_horizon) # type: ignore

            # Añadimos los resultados a las listas correspondientes:
            predictions.append(pred) # type: ignore
            errors.append(new_pipeline.score(val, pred)) # type: ignore
            # Y prograsamos el versor de entrenamiento:
            if verbose:
                print(f"SMAPE {start}: {errors[-1]:.2f}")
            start += step
            
        # Devolvemos el SMAPE promedio y las predicciones concatenadas:
        return np.mean(errors), concatenate(predictions) # type: ignore