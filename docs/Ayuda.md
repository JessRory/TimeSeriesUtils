# 📚 Documentación de Librería: `TimeSeriesFiles`

**Versión:** 0.0.2

**Dependencias:** `pandas`, `tkinter`, `rapidfuzz`, `darts`, `scikit learn`, `statsmodels`, `scypy`, `matplotlib`

---

# 📈 TimeSeriesFiles: Framework de Forecasting con Darts

Este paquete proporciona una arquitectura modular y profesional para el análisis y predicción de series temporales, construida sobre la librería **Darts**. Está diseñado para facilitar el flujo de trabajo desde la conversión de datos crudos hasta la validación rigurosa de modelos.

## 📂 Estructura del Proyecto

```text
TimeSeriesFiles/
├── utilities.py       # Clases core: Pipeline y Conversión de datos
├── Ayuda.md           # Documentación del paquete (este archivo)
└── utils/
    ├── analisis.py    # Herramientas de diagnóstico y descomposición
    └── evaluators.py  # Backtesting y métricas de error

```

---

## 🚀 Componentes Principales

### 1. `DartsPipeline` (en `utilities.py`)

Es el motor del paquete. Permite encadenar un modelo de predicción con un transformador (como un `Scaler`) de forma transparente.

* **Gestión inteligente:** Si no se necesita escalado (como en modelos clásicos), el pipeline lo detecta y procesa la serie original.
* **Consistencia:** Asegura que cualquier transformación aplicada al entrenar se revierta automáticamente al predecir (`inverse_transform`).

### 2. `SeriesConverter` (en `utilities.py`)

El puente entre Pandas y Darts.

* **Robustez:** Valida tipos de datos, gestiona duplicados y asegura la continuidad de la frecuencia temporal.
* **Frecuencia Estricta:** Fuerza la frecuencia (ej. 'QS' para trimestres) para evitar los errores comunes de "Missing Frequency" en Darts.

### 3. `TimeSeriesAnalyzer` (en `utils/analisis.py`)

Herramienta de diagnóstico visual.

* **Descomposición Estacional:** Separa la serie en **Tendencia, Estacionalidad y Residuo** usando un enfoque aditivo.
* **Detección de Frecuencia:** Ayuda a identificar el patrón de muestreo del dataset original.

### 4. `TimeSeriesEvaluator` (en `utils/evaluators.py`)

La pieza clave para la confianza del modelo.

* **Walk-Forward Backtest:** Simula el paso del tiempo re-entrenando el modelo progresivamente. Es la forma más realista de medir el error (sMAPE) antes de ir a producción.

---

# 1. Módulo de Diagnosis: `diagnostico_serie`

Este método estático es la herramienta fundamental de la **Fase 0**. Su objetivo es eliminar las "conjeturas" al configurar un modelo de Darts, permitiéndote elegir parámetros basados en evidencia estadística.

## 1.1. El Código:

Este método se define como `@staticmethod` porque es una función de utilidad que no necesita modificar el estado de la clase, solo procesa los datos que recibe.

```python
@staticmethod
def diagnostico_serie(df: pd.DataFrame, valor_col: str, freq_anual: int = 12) -> None:
    # ... (código)

```

## 1.2. Guía de Interpretación:

Cada uno de los 6 gráficos tiene una misión directa para configurar `DartsPipeline` o `ForecastingModel`:

### 1.2.1. Gráfico de Tendencia: ¿Necesito `Detrender`?

* **Qué mirar:** Si la línea sube o baja de forma sostenida (como en la población).
* **Acción en Darts:** Si hay tendencia, el test ADF (punto 6) lo confirmará. Se deberá incluir `Detrender(model=LinearRegression())` en la Pipeline.

### 1.2.2. Boxplot Estacional: ¿La variabilidad es constante?

* **Qué mirar:** Si las "cajas" de los meses tienen tamaños muy diferentes.
* **Acción en Darts:** Si la variabilidad cambia con el tiempo, el uso de un `Scaler()` es obligatorio para normalizar los rangos antes de que el modelo aprenda.

### 1.2.3. ACF (Autocorrelación): Confirmar Estacionalidad

* **Qué mirar:** Picos que sobresalen significativamente en múltiplos de la frecuencia (ej. en el lag 12, 24).
* **Acción en Darts:** Te confirma que la serie tiene memoria estacional. Ayuda a decidir el horizonte de predicción.

### 1.2.4. PACF (Autocorrelación Parcial)  El parámetro `lags`

* **Qué mirar:** ¿En qué número de lag las barras dejan de ser significativas (entran en la zona sombreada)?
* **Acción en Darts:** Este es el valor ideal para el parámetro `lags=`.
* *Ejemplo:* Si solo los primeros 3 lags son significativos, usa `lags=3`. Meter más solo añadirá ruido.

### 1.2.5. Periodograma  El parámetro `fourier`

* **Qué mirar:** Los picos más altos en el espectro de frecuencias.
* **Acción en Darts:** Si ves un pico claro, activa los encoders de Fourier.
* *Ejemplo:* `add_encoders={'fourier': {'future': {'month': 4}}}`. El número de armónicos (4) dependerá de lo complejo que sea el pico en este gráfico.

### 1.2.6. Test ADF  Validación Final de Estacionariedad

* **Qué mirar:** El **p-value**.
* **p < 0.05:** La serie es estacionaria.
* **p > 0.05:** La serie NO es estacionaria (tiene tendencia o raíz unitaria).

* **Acción en Darts:** Si es > 0.05, el `Detrender` en el Pipeline no es opcional, es **necesario**.

### 1.2.7. Uso de Covariables (PIB, IPC, etc.):

* **Sincronización:** Asegúrate de que la serie del PIB empiece en la misma fecha que la de población y cubra todo el horizonte de predicción (n).
* **Escalado:** Escala el PIB de forma independiente a la población antes de pasarlo al fit.
* **Inyección:** Pasa la serie del PIB a través del argumento future_covariates tanto en el método fit() como en el predict() de tu DartsPipeline.
* **Modelos compatibles:** Recuerda que modelos como XGBModel, TFTModel y RegressionModel aprovechan muy bien estas variables. Modelos clásicos como ARIMA (el estándar) no las usan de esta forma (necesitarías AutoARIMA o VARIMA).

---

## 2. Análisis de datos por Boxplot: Módulo de Visualización: `anotar_boxplot`

Este método estático permite realizar una «disección» visual de la distribución de la serie temporal, facilitando la identificación de rangos intercuartílicos y valores atípicos (outliers) sin necesidad de consultar tablas de datos.

### 2.1. Documentación Técnica (Docstring)
Python

    @staticmethod
    def anotar_boxplot(df: pd.DataFrame, x: str, y: str, ax: plt.Axes = None) -> plt.Axes:
        """
        Dibuja un boxplot enriquecido con anotaciones estadísticas automáticas.
        
        Calcula y etiqueta visualmente la Mediana, los Cuartiles (Q1, Q3) y los 
        Bigotes (mínimo/máximo no atípico), además de incluir un indicador 
        del Rango Intercuartílico (IQR).
        
        Parámetros:
        - df: pd.DataFrame, el conjunto de datos que contiene la serie.
        - x: str, nombre de la columna para el eje X (ej: 'periodo').
        - y: str, nombre de la columna para el eje Y (ej: 'valor').
        - ax: plt.Axes, opcional. Eje de Matplotlib donde dibujar. 
          Si es None, crea uno nuevo.
        
        Devuelve:
        - plt.Axes: El objeto de los ejes con todas las anotaciones aplicadas.
        """
### 2.2. Guía de Interpretación:

Caja Central (Q1 a Q3): Representa el 50% central de tus datos. Si la caja es muy "alta", la población tiene mucha variabilidad en esos periodos; si es "chata", los valores son muy estables.

Línea de Mediana: Si no está en el centro de la caja, indica que los datos tienen asimetría. Esto sugiere que quizás necesitemos un Scaler robusto en DartsPipeline.

Indicador IQR (Flecha Roja): Es el Rango Intercuartílico y es vital para entender la dispersión. Un IQR que cambia mucho entre diferentes periodos (ej. entre meses) sugiere que se deben usar modelos que capturen la estacionalidad de la varianza.

Puntos fuera de Bigotes: Son valores atípicos. Si hay muchos, hay que considerar si es necesario un paso previo de limpieza o si el modelo debe ser más complejo para no "asustarse" con esos picos.

Tip de Pro
:
Como esta función devuelve un plt.Axes, se puede encadenar con otras. Por ejemplo, puedes llamar a anotar_boxplot y después añadirle un título personalizado o cambiar los colores del fondo:

Python

ax = analizer.anotar_boxplot(df, 'mes', 'poblacion')
ax.set_title("Análisis de Dispersión Mensual")

---

## 3. Módulo de Diagnóstico Estadístico: `test_aditiva_multiplicativa`:

Este método estático es el "árbitro" técnico que decide la naturaleza de la serie temporal. Su función es determinar si la estacionalidad de la serie es constante (Aditiva) o si escala proporcionalmente con la tendencia (Multiplicativa).

### 3.1. Documentación Técnica (Docstring):

Python

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
### 3.2. Guía de Aplicación en el Modelado:

Si el resultado es ADITIVA:

Interpretación: La magnitud de los ciclos estacionales (ej. picos de verano) es similar cada año, sin importar si la población del estudio sube o baja.

Acción en Darts: Puedes usar el Detrender estándar y modelos lineales o basados en árboles (XGBModel) con los datos en su escala original.

Si el resultado es MULTIPLICATIVA:

Interpretación: Las variaciones estacionales se "estiran" a medida que la serie crece. Típico en crecimientos exponenciales o por tasas.

Acción en Darts: 
1. Transformación: Es altamente recomendable usar un LogTransformer en DartsPipeline para linealizar la serie. 
2. Modelos Específicos: Si usas ExponentialSmoothing, asegúrate de configurar seasonal=SeasonalityMode.MULTIPLICATIVE.

### 3.3. Interpretación Visual del Gráfico de Residuos:

Cuando mostrar_grafico=True, busca lo siguiente:

Ruido Blanco: El residuo ideal debe verse como una banda de puntos aleatorios centrada en cero.

Efecto Embudo: Si en el residuo aditivo ves que la dispersión aumenta al final del tiempo (forma de abanico), es una confirmación visual de que la serie es Multiplicativa.

---

## 📝 Resumen::

> "Antes de instanciar cualquier modelo en Darts, ejecuta `diagnostico_serie`.
> 1. Usa el **PACF** para fijar los `lags`.
> 2. Usa el **Periodograma** para decidir si aplicar `fourier`.
> 3. Usa el **ADF** para decidir si tu `DartsPipeline` requiere un `Detrender`.
> 

"Si vas a usar XGBModel, RandomForestModel o TFTModel, recuerda que la magia ocurre al combinar:

El Detrender en el Pipeline (para que el modelo trabaje en una serie plana).

Los lags (basados en tu análisis PACF).

El add_encoders con Fourier (basado en tu análisis de Periodograma)."

> Un buen análisis previo reduce el error (SMAPE) drásticamente al evitar que el modelo intente aprender patrones inexistentes."

---

## 🛠️ Guía de uso rápido:

Una vez examinada la serie, elegido el modelo y aclarados los parámetros:

### Paso 1: Convertir tus datos:

```python
from TimeSeriesFiles.utilities import SeriesConverter

serie = SeriesConverter.convertir_a_TimeSeries(
    df=mi_dataframe, 
    time_col='fecha', 
    datos_col='valor', 
    freq='QS', 
    fillna=False
)

```

### Paso 2: Analizar la estructura:

```python
from TimeSeriesFiles.utils.analisis import TimeSeriesAnalyzer

# Analizamos estacionalidad anual (4 trimestres)
TimeSeriesAnalyzer.analizar_serie(serie, freq_estacional=4)

```

### Paso 3: Validar y Predecir:

```python
from darts.models import ExponentialSmoothing
from TimeSeriesFiles.utilities import DartsPipeline
from TimeSeriesFiles.utils.evaluators import TimeSeriesEvaluator

# Configurar Pipeline
modelo = ExponentialSmoothing(seasonal_periods=4)
pipeline = DartsPipeline(model=modelo)

# Backtesting para medir error real
smape_medio, preds_hist = TimeSeriesEvaluator.walk_forward_backtest(
    pipeline=pipeline, 
    series=serie, 
    forecast_horizon=4
)

# Predicción final al futuro
pipeline.fit(serie)
pronostico = pipeline.predict(n=8)

```

---

## 💡 Notas Técnicas (Decisiones de Diseño):

* **Escalado de Datos:** En este framework, el escalado es opcional. Para modelos **clásicos** (ARIMA, Exponential Smoothing), se recomienda usar `transformer=None` para mantener la interpretabilidad y escala real. Para modelos de **Machine Learning / Deep Learning**, es imperativo pasar un `Scaler()` al `DartsPipeline`.
* **Métrica sMAPE:** Se utiliza el *Symmetric Mean Absolute Percentage Error* por ser una métrica acotada (0-200%) que trata por igual las sobreestimaciones y las subestimaciones.