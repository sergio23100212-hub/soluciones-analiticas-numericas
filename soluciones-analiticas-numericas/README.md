# Soluciones Analíticas y Numéricas

Experimentos reproducibles de ecuaciones diferenciales con Python y reportes JSON.

## Instalación

```bash
python -m pip install -r requirements.txt
python src/experiments.py
```

Se generan `reports/convergence.json`, `reports/stability.json`, `reports/laplace.json` y `reports/manifest.json`.

- **Convergencia:** Euler y RK4 contra la solución exacta de `y'=-2y`, con diferentes pasos.
- **Estabilidad:** valores propios y trayectoria de un sistema lineal de dos variables.
- **Laplace:** solución simbólica y comprobación de una EDO de segundo orden.

Los tiempos en nanosegundos son mediciones de una sola ejecución; no constituyen benchmarks estadísticos. Los métodos numéricos no sustituyen la validación del modelo físico.
