# 📚 Documentación de Librería: `TimeSeriesFiles`

**Versión:** 0.0.1

**Dependencias:** `pandas`, `tkinter`, `rapidfuzz`, `darts`, `scikit learn`, `statsmodels`

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

## 🛠️ Guía de Uso Rápido

### Paso 1: Convertir tus datos

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

### Paso 2: Analizar la estructura

```python
from TimeSeriesFiles.utils.analisis import TimeSeriesAnalyzer

# Analizamos estacionalidad anual (4 trimestres)
TimeSeriesAnalyzer.analizar_serie(serie, freq_estacional=4)

```

### Paso 3: Validar y Predecir

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

## 💡 Notas Técnicas (Decisiones de Diseño)

* **Escalado de Datos:** En este framework, el escalado es opcional. Para modelos **clásicos** (ARIMA, Exponential Smoothing), se recomienda usar `transformer=None` para mantener la interpretabilidad y escala real. Para modelos de **Machine Learning / Deep Learning**, es imperativo pasar un `Scaler()` al `DartsPipeline`.
* **Métrica sMAPE:** Se utiliza el *Symmetric Mean Absolute Percentage Error* por ser una métrica acotada (0-200%) que trata por igual las sobreestimaciones y las subestimaciones.