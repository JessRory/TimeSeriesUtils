Markdown

# 📈 Framework de Series Temporales con Darts

Este repositorio contiene una arquitectura modular en Python para el análisis, procesamiento y predicción de series temporales utilizando la librería **Darts**. 

Fue diseñado originalmente para predecir la población de España con datos oficiales del INE, logrando una precisión con un error (sMAPE) inferior al 0.20%.

## 📂 Estructura del Repositorio

* **`TimeSeriesFiles/`**: Paquete principal con la lógica de negocio.
    * `utilities.py`: Contiene el `DartsPipeline` y el `SeriesConverter`.
    * `utils/analisis.py`: Herramientas para descomposición estacional y diagnóstico.
    * `utils/evaluators.py`: Implementación de *Backtesting* (Walk-forward).
* **`Ayuda.md`**: Guía detallada de uso y documentación técnica de las clases.
* **`LICENSE`**: Licencia MIT.

## 🚀 Inicio Rápido

```python
from TimeSeriesFiles.utilities import SeriesConverter, DartsPipeline
from darts.models import ExponentialSmoothing

# 1. Convertir datos de Pandas a Darts
serie = SeriesConverter.convertir_a_TimeSeries(df, 'fecha', 'poblacion', freq='QS')

# 2. Configurar y entrenar el modelo
modelo = ExponentialSmoothing(seasonal_periods=4)
pipeline = DartsPipeline(model=modelo)
pipeline.fit(serie)

# 3. Predecir el futuro
prediccion = pipeline.predict(n=4)
🧠 Decisiones Técnicas
Modelos Clásicos: Se prioriza el uso de modelos como Exponential Smoothing para datos con tendencia clara y baja volatilidad.

Sin Escalado: Para mantener la interpretabilidad de los datos demográficos, el pipeline trabaja con las magnitudes reales.

Validación Rigurosa: Se incluye un evaluador de avance progresivo para garantizar que el modelo sea fiable antes de realizar predicciones futuras.

Distribuido bajo la Licencia MIT.
