import pandas as pd # type: ignore
from darts import TimeSeries # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore
import matplotlib.pyplot as plt # type: ignore

class TimeSeriesAnalyzer:
    """
    Herramientas de diagnóstico y visualización para series temporales.
    """
    
    @staticmethod
    def analizar_serie(series: TimeSeries, freq_estacional: int) -> None:
        """
        Descompone la serie en Tendencia, Estacionalidad y Residuo.
        Parámetros:
        - series: TimeSeries
        - freq_estacional: int, frecuencia de la estacionalidad (ej: 12 para mensual, 7 para diario)
        
        Devuelve: None, pero muestra gráficas de la descomposición.
        """
        # Convertimos a Pandas para la compatibilidad con statsmodels
        df_temp = series.to_dataframe() # type: ignore
        
        # Aplicamos la descomposición clásica
        # period: frecuencia de la estacionalidad (ej: 12 para mensual, 7 para diario)
        descomposicion = seasonal_decompose(df_temp, period=freq_estacional) # type: ignore
        
        # Configuración de la gráfica
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True) # type: ignore
        
        # Serie Original
        axes[0].set_title("Serie temporal original")
        df_temp.plot(ax=axes[0], color='blue') # type: ignore
        
        # Tendencia
        axes[1].set_title("Tendencia (Trend)")
        descomposicion.trend.plot(ax=axes[1], color='red') # type: ignore
        
        # Estacionalidad
        axes[2].set_title(f"Estacionalidad (Seasonality - Period: {freq_estacional})")
        descomposicion.seasonal.plot(ax=axes[2], color='green') # type: ignore
        
        # Residuos (Ruido)
        axes[3].set_title("Residuos (Residuals/Noise)")
        descomposicion.resid.plot(ax=axes[3], color='gray', kind='kde') # KDE ayuda a ver si el ruido es normal # type: ignore
        
        plt.tight_layout()
        plt.show() # type: ignore
     
    @staticmethod   
    def detectar_frecuencia_ts(df:pd.DataFrame, columna:str)-> None:
        
        """
        Detecta la frecuencia más común en una columna de un DataFrame.
        
        Parámetros:
        - df: pd.DataFrame, DataFrame que contiene la columna a analizar.
        - columna: str, nombre de la columna a analizar.
        
        Devuelve: None, pero imprime la frecuencia más común.
        """
        
        frec = df[columna].diff().value_counts().idxmax()
        print(f"La frecuencia más común en la columna '{columna}' es: {frec}")
        
    