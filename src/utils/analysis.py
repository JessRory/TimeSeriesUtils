import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from darts import TimeSeries # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore
from statsmodels.tsa.stattools import adfuller # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # type: ignore
from scipy import signal # type: ignore
import statsmodels.api as sm # type: ignore
from statsmodels.regression.linear_model import RegressionResultsWrapper # type: ignore

class TimeSeriesAnalyzer:
    """
    Herramientas de diagnóstico y visualización para series temporales.
    """
    
    @staticmethod
    def analizar_serie(series: TimeSeries, freq_estacional: int) -> None: #type: ignore
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
        

    @staticmethod
    def diagnostico_serie(df: pd.DataFrame, valor_col: str, freq_anual: int = 12)-> None:
        """
        Realiza un diagnóstico completo para configurar modelos en Darts.
        Parámetros:
        
        df: DataFrame de pandas con índice temporal.
        valor_col: Nombre de la columna con los datos.
        freq_anual: 12 para mensual, 4 para trimestral, 365 para diario.
        
        Devuelve: None, pero muestra gráficas y recomendaciones.
        """
        fig = plt.figure(figsize=(15, 12)) # type: ignore
        plt.subplots_adjust(hspace=0.4)
        
        # 1. Gráfico de Línea (Tendencia):
        ax1 = plt.subplot(3, 2, 1) # type: ignore
        df[valor_col].plot(ax=ax1, color='navy')
        ax1.set_title("1. Gráfico de tendencia") # type: ignore
        
        # 2. Boxplot Estacional:
        ax2 = plt.subplot(3, 2, 2) # type: ignore
        # Suponiendo que el índice es datetime
        temp_df = df.copy()
        temp_df['periodo'] = df.index.month if freq_anual == 12 else df.index.quarter # type: ignore
        sns.boxplot(data=temp_df, x='periodo', y=valor_col, ax=ax2)
        ax2.set_title("2. Variabilidad Estacional (Boxplot)") # type: ignore

        # 3. ACF (Autocorrelación) -> Para estacionalidad
        ax3 = plt.subplot(3, 2, 3) # type: ignore
        plot_acf(df[valor_col], lags=freq_anual*2, ax=ax3)
        ax3.set_title("3. ACF (Autocorrelación)") # type: ignore

        # 4. PACF (Autocorrelación Parcial) -> PARA EL PARÁMETRO 'lags'
        ax4 = plt.subplot(3, 2, 4) # type: ignore
        plot_pacf(df[valor_col], lags=freq_anual*2, ax=ax4, method='yule_walker')
        ax4.set_title("4. PACF (Autocorrelación Parcial)") # type: ignore

        # 5. Periodograma (Espectro) -> PARA EL PARÁMETRO 'fourier'
        ax5 = plt.subplot(3, 2, 5) # type: ignore
        f, Pxx_den = signal.periodogram(df[valor_col].fillna(method='ffill').fillna(method='bfill').fillna(0)) # type: ignore
        ax5.semilogy(f, Pxx_den)# type: ignore
        ax5.set_title("5. Periodograma (Picos = ciclos dominantes)") # type: ignore
        ax5.set_xlabel("Frecuencia") # type: ignore
        ax5.set_ylabel("Densidad espectral de potencia") # type: ignore
        
        # 6. Test ADF (Estacionariedad)
        ax6 = plt.subplot(3, 2, 6) # type: ignore
        result = adfuller(df[valor_col].fillna(method='ffill').fillna(method='bfill').fillna(0)) # type: ignore
        ax6.text(0.1, 0.5, f"ADF Statistic: {result[0]:.4f}\np-value: {result[1]:.4f}", # type: ignore
                fontsize=14, va='center') 
        ax6.axis('off')
        ax6.text(0.1, 0.5, f"ADF Statistic: {result[0]:.4f}\np-value: {result[1]:.4f}"), # type: ignore
        ax6.set_title("6. Test de Estacionariedad (ADF)") # type: ignore

        plt.show() # type: ignore

        # Recomendación rápida por consola
        print("--- GUÍA DE CONFIGURACIÓN DARTS ---")
        if result[1] > 0.05:
            print("-> ADF p-value > 0.05: Serie NO estacionaria. Usa Detrender o Diff() en tu Pipeline.")
        else:
            print("-> Serie Estacionaria. Puedes usar modelos sin transformación previa.")
        
        print(f"-> Mira el gráfico 4: El último lag significativo es el candidato para 'lags='.")
        print(f"-> Mira el gráfico 5: Si hay picos claros, usa 'add_encoders' con fourier.")
        
    @staticmethod
    def anotar_boxplot(df: pd.DataFrame, x: str, y: str, ax=None) -> plt.Axes: # type: ignore
        """
        Dibuja un boxplot con etiquetas de texto alineadas a la derecha.
        
        Parámetros:
        - df: DataFrame de pandas con los datos.
        - x: Nombre de la columna para el eje X (categorías).
        - y: Nombre de la columna para el eje Y (valores numéricos).
        - ax: Eje de Matplotlib opcional para dibujar el boxplot.
        
        Devuelve: Eje de Matplotlib con el boxplot anotado.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6)) # type: ignore

        # Dibujamos el boxplot base
        sns.boxplot(data=df, x=x, y=y, ax=ax, color='#1f77b4') # type: ignore
        
        # 1. Obtenemos los estadísticos
        group_data = df[y].dropna()
        q1, mediana, q3 = np.percentile(group_data, [25, 50, 75])
        
        # 2. Añadimos el texto a la derecha:
        # Usamos coordenadas relativas al eje X (0.26 suele ser justo al lado de la caja):
        ax.text(0.26, q3, f'Q3 ({q3:.2f})', va='center', ha='left', fontsize=10) # type: ignore
        ax.text(0.26, mediana, f'Mediana ({mediana:.2f})', va='center', ha='left', fontsize=10) # type: ignore
        ax.text(0.26, q1, f'Q1 ({q1:.2f})', va='center', ha='left', fontsize=10) # type: ignore

        # 3. Calculamos Bigotes para las etiquetas superiores/inferiores
        iqr_val = q3 - q1
        v_max = group_data[group_data <= q3 + 1.5 * iqr_val].max()
        v_min = group_data[group_data >= q1 - 1.5 * iqr_val].min()
        
        ax.text(-0.4, v_max, f'Bigote superior ({v_max:.2f})', va='center', ha='left', fontsize=10) # type: ignore
        ax.text(-0.4, v_min, f'Bigote inferior ({v_min:.2f})', va='center', ha='left', fontsize=10) # type: ignore

        # 4. Indicador del IQR (La flecha roja lateral)
        ax.annotate('', # type: ignore
                    xy=(0.8, q1), 
                    xytext=(0.8, q3),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        
        ax.text(0.82, (q1 + q3) / 2, f'IQR: {iqr_val:.2f}', color='red', va='center', fontweight='bold') # type: ignore

        # Estética final
        ax.yaxis.grid(True, linestyle='--', alpha=0.6) # type: ignore
        
        return ax # type: ignore
    
    @staticmethod
    def test_aditiva_multiplicativa(df: pd.DataFrame, valor_col: str, freq: int = 12, mostrar_grafico: bool = False) -> str:
        """
        Determina estadísticamente si una serie es Aditiva o Multiplicativa.
        Parámetros:
        - df: DataFrame de pandas con los datos.
        - valor_col: Nombre de la columna con los valores de la serie.
        - freq: Frecuencia de la serie (12 para mensual, 4 para trimestral, etc.)
        - mostrar_grafico: Si es True, muestra un gráfico comparativo de residuos. 
        Por defecto es False.
        
        Devuelve:
        - str: 'additive' o 'multiplicative' según el mejor ajuste.
        """
        
        # 1. Preparación y limpieza (evitamos nulos para statsmodels)
        serie = df[valor_col].ffill().bfill() # type: ignore
        
        # El modelo multiplicativo requiere valores estrictamente positivos (> 0)
        if (serie <= 0).any(): # type: ignore
            serie = serie - serie.min() + 1.0 # type: ignore

        # 2. Descomposición
        res_adi = seasonal_decompose(serie, model='additive', period=freq) # type: ignore
        res_mult = seasonal_decompose(serie, model='multiplicative', period=freq) # type: ignore

        # 3. Métrica: Calculamos la estabilidad del residuo
        # Un residuo con menor desviación estándar relativa indica mejor modelo
        std_adi = np.nanstd(res_adi.resid) / np.nanmean(serie) # type: ignore
        std_mult = np.nanstd(res_mult.resid) / np.nanmean(serie) # type: ignore

        # 4. Decisión
        recomendacion = 'Serie aditiva' if std_adi < std_mult else 'Serie multiplicativa'
        
        # 5. Visualización (Opcional, pero muy útil para el cuaderno)
        if mostrar_grafico:
            plt.figure(figsize=(12, 5)) # type: ignore
            plt.plot(res_adi.resid, label=f'Residuo Aditivo (Rel. Std: {std_adi:.4f})', alpha=0.7) # type: ignore
            plt.plot(res_mult.resid, label=f'Residuo Multiplicativo (Rel. Std: {std_mult:.4f})', alpha=0.7) # type: ignore
            plt.title(f"Comparación de Residuos - Recomendación: {recomendacion.upper()}") # type: ignore
            plt.legend() # type: ignore
            plt.show() # type: ignore

        return recomendacion
    
    @staticmethod
    def plot_periodograma(series:pd.Series,
                          fs:int=1,
                          detrend:str='linear',
                          vertical_lines:bool=False)->tuple[np.ndarray[()], float]:
        """
        Analiza las frecuencias dominantes de una serie temporal.
        
        Parámetros:
        - series: Array-like o Pandas Series.
        - fs: Frecuencia de muestreo (ej: 1 para diario, 1/7 para semanal si la unidad es días).
        - detrend: 'linear', 'constant' o None. Ayuda a ver ciclos en series con tendencia.
        - vertical_lines: Diccionario {Nombre: Frecuencia} para marcar puntos de interés.
        
        Devuelve: Tuple con frecuencias y densidad espectral de potencia.
        """
        # Convertimos a numpy y limpiamos nulos si los hay
        data = np.asarray(series)
        data = data[~np.isnan(data)]
        
        # Eliminamos tendencia para resaltar periodicidad
        if detrend:
            data = signal.detrend(data, type=detrend) # type: ignore
        
        f, Pxx_den = signal.periodogram(data, fs=fs) # type: ignore

        plt.figure(figsize=(12, 5)) # type: ignore
        plt.semilogy(f, Pxx_den, color='black', alpha=0.8) # type: ignore
        
        # Dibujamos líneas de interés si el usuario las pasa
        if vertical_lines:
            colors = plt.cm.get_cmap('viridis', len(vertical_lines)) # type: ignore
            for i, (label, freq) in enumerate(vertical_lines.items()): # type: ignore
                plt.axvline(freq, color=colors(i), linestyle='--', label=label) # type: ignore
            plt.legend() # type: ignore

        plt.title('Análisis de Densidad Espectral de Potencia (Periodograma)') # type: ignore
        plt.xlabel('Frecuencia (ciclos por unidad de tiempo)') # type: ignore
        plt.ylabel('Potencia') # type: ignore
        plt.grid(True, which="both", alpha=0.3) # type: ignore
        plt.show() # type: ignore
        
        return f, Pxx_den # type: ignore
    
    @staticmethod
    def analizar_fourier(series_pandas:pd.Series, n_armonicas:int=3, period:int=52)->RegressionResultsWrapper: # type: ignore
        """
        Calcula y grafica la estacionalidad usando términos de Fourier (Seno/Coseno).
        Ideal para confirmar ciclos como las 'vacaciones de verano'.
        
        Parámetros:
        - series_pandas: Serie temporal en formato Pandas Series.
        - n_armonicas: Número de armónicas a incluir (más = más detalle, pero riesgo de overfitting).
        - period: Periodo de la estacionalidad (ej: 52 para semanal con ciclo anual).
        
        Devuelve: El modelo ajustado de Fourier. Este contiene:
        model.params: Un diccionario/serie con los coeficientes calculados para cada onda de seno y coseno (las amplitudes).
        model.pvalues: Los valores «p» para saber si cada armónico es estadísticamente significativo.
        model.rsquared: El R², que señala qué porcentaje de la variación de Bitcoin explica esa estacionalidad de Fourier.
        model.predict(): El método para generar valores futuros de esa misma onda.
        Esto permite, por ejemplo:
        
        Guardamos el resultado del análisis:
            resultado = TimeSeriesAnalyzer.analizar_fourier(serie_temporal, n_armonicas=3)

        Inspeccionar los detalles técnicos sin volver a calcular nada:
            print(f"Precisión del ajuste (R2): {resultado.rsquared:.4f}")

        Ver qué armónico es el más importante
            print(resultado.params)
        """

        # 1. Preparación de datos (aseguramos que sea un array de numpy)
        y = series_pandas.values
        t = np.arange(len(y))
        X = pd.DataFrame(index=series_pandas.index)
        
        # 2. Generación de las armónicas
        for k in range(1, n_armonicas + 1):
            X[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
            X[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
        
        # 3. Ajuste del modelo OLS
        X_with_const = sm.add_constant(X) # type: ignore
        model = sm.OLS(y, X_with_const).fit() # type: ignore 
        
        # 4. Extracción de componentes
        prediccion_total = model.predict(X_with_const) # type: ignore
        # La estacionalidad es la suma de los efectos seno/coseno
        estacionalidad_pura = prediccion_total - model.params['const'] # type: ignore
        
        # 5. Visualización
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # type: ignore
        
        # Panel superior: Serie vs Ajuste
        ax1.plot(series_pandas.index, y, label='Serie Real (Log)', color='gray', alpha=0.4)
        ax1.plot(series_pandas.index, prediccion_total, label='Tendencia Base + Fourier', color='red', lw=2)
        ax1.set_title(f'Ajuste de Fourier sobre la Serie (k={n_armonicas} armónicas)')
        ax1.legend()
        
        # Panel inferior: La "Onda" estacional pura
        ax2.plot(series_pandas.index, estacionalidad_pura, label='Ciclo Estacional Teórico', color='blue', lw=2)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Componente Estacional Extraído (Seno/Coseno)')
        ax2.set_ylabel('Amplitud (Influencia en el Precio)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show() # type: ignore
        
        return model # type: ignore