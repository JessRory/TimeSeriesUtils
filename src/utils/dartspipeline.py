import pandas as pd # type: ignore
from darts import TimeSeries # type: ignore
from darts.metrics import smape # type: ignore
from typing import Optional, Self
from darts.models.forecasting.forecasting_model import ForecastingModel # type: ignore
from darts.dataprocessing.transformers import FittableDataTransformer # type: ignore

class DartsPipeline:
    """
    Pipeline simple para modelado de series temporales usando Darts.
    Combina un modelo de forecasting con un transformador de datos opcional.
    """
    def __init__(self, model: ForecastingModel, transformer: Optional[FittableDataTransformer]= None) -> None:
        """
        Inicializa la Pipeline.
        
        Parámetros:
        - model: ForecastingModel, modelo de forecasting de Darts.
        - transformer: Optional[FittableDataTransformer], transformador de datos de Darts (ej: Scaler).
        Si es None, no se aplica transformación.
        
        Devuelve: None
        """
        # Instanciamos los componentes:
        self.model = model
        self.transformer = transformer
        self.series_original = None
        self.series_transformed = None
        self.is_fitted = False

    def fit(self, series: TimeSeries) -> Self:
        """
        Ajusta la Pipeline a la serie temporal dada.
        
        Parámetros:
        - series: TimeSeries, serie temporal para ajustar la Pipeline.

        Devuelve: Self, instancia de la Pipeline ajustada.
        """
        # Entrenamos el transformador (si existe) y luego el modelo:
        self.series_original = series
        self.series_transformed = self.transformer.fit_transform(series) if self.transformer else series # type: ignore
        self.model.fit(self.series_transformed) # type: ignore
        self.is_fitted = True
        return self

    def predict(self, n: int) -> TimeSeries:
        """
        Predice los próximos n pasos.
        
        Parámetros:
        - n: int, número de pasos a predecir.
        
        Devuelve: TimeSeries, serie temporal con las predicciones.
        """
        # Predecimos con el modelo entrenado y revertimos la transformación (si existe):
        if not self.is_fitted:
            raise RuntimeError("Debes llamar a `fit()` antes de `predict()`.")
        forecast = self.model.predict(n)
        
        # Devolvemos la predicción inversamente transformada (si aplica):
        return self.transformer.inverse_transform(forecast) if self.transformer else forecast # type: ignore

    def fit_predict(self, series: TimeSeries, n: int) -> TimeSeries:
        """
        Ajusta la Pipeline a la serie temporal dada y predice los próximos n pasos.
        
        Parámetros:
        - series: TimeSeries, serie temporal para ajustar la Pipeline.
        - n: int, número de pasos a predecir.
        
        Devuelve: TimeSeries, serie temporal con las predicciones.
        """
        # Entrenamos y predecimos al mismo tiempo usando los métodos anteriores:
        return self.fit(series).predict(n)

    def score(self, actual: TimeSeries, forecast: TimeSeries) -> float:
        """
        Calcula el SMAPE entre la serie real y la predicción.
        
        Parámetros:
        - actual: TimeSeries, serie temporal real.
        - forecast: TimeSeries, serie temporal predicha.
        
        Devuelve: float, valor del SMAPE.
        """
        # Calculamos el SMAPE usando la función de Darts:
        return smape(actual, forecast) # type: ignore

class SeriesConverter:
    """
    Clase dedicada a la transformación de estructuras de datos 
    para compatibilidad con la librería Darts.
    """
    
    @staticmethod
    def convertir_a_TimeSeries(
        df: pd.DataFrame, 
        time_col: str, 
        datos_col: str, 
        freq: str,
        fillna: bool = False 
    ) -> TimeSeries:
        """
        Convierte un DataFrame de Pandas a un objeto TimeSeries de Darts.
        
        Args:
            df: DataFrame de entrada.
            time_col: Nombre de la columna con las fechas.
            datos_col: Nombre de la columna con el valor a predecir.
            freq: Código de frecuencia de Pandas (e.g., 'QS', 'MS', 'D').
            fillna: Si es True, rellena huecos temporales para evitar errores.
        """
        # 1. Copia y limpieza básica para no mutar el original
        temp_df = df[[time_col, datos_col]].copy()
        
        # 2. Asegurar que la columna es datetime y quitar duplicados
        if not pd.api.types.is_datetime64_any_dtype(temp_df[time_col]):
            print(f"ℹ️ La columna '{time_col}' no es datetime. Convirtiendo...")
            temp_df[time_col] = pd.to_datetime(temp_df[time_col])
            
        temp_df = temp_df.drop_duplicates(subset=[time_col])
        
        # 3. Establecer índice y forzar frecuencia (Crucial para Darts)
        temp_df.set_index(time_col, inplace=True)
        temp_df = temp_df.asfreq(freq)
        
        # 4. Gestión de huecos (si los hay)
        if fillna and temp_df[datos_col].isnull().any():
            print(f"ℹ️ Detectados valores nulos en la serie. Aplicando 'forward fill'.")
            temp_df[datos_col] = temp_df[datos_col].ffill()
            
        # 5. Creación del objeto TimeSeries
        series = TimeSeries.from_dataframe( #type: ignore
            temp_df.reset_index(), 
            time_col=time_col, 
            value_cols=datos_col, 
            freq=freq
        )
        
        return series