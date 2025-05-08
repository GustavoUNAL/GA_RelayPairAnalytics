# GA\_RelayPairAnalytics

> *Análisis integral de la coordinación de pares de relés IEC mediante Python*

---

## Descripción

Repositorio para procesar y visualizar los resultados de optimización de pares de relés (pickup & TDS) generados por algoritmos heurísticos y genéticos. Incluye utilidades para calcular métricas de coordinación (MT, TMT, CTI cumplido/incumplido), contrastar ajustes iniciales vs. finales y producir reportes/figuras listos para tesis o artículos.

## Estructura sugerida

```text
.
├─ data/
│  ├─ raw/                      # JSON de entrada (pares de relés por escenario)
│  └─ processed/                # Archivos optimizados (salida MATLAB / GA)
├─ notebooks/
│  └─ analysis_pairs.ipynb      # Ejemplo interactivo de análisis
├─ relay_analysis/
│  ├─ __init__.py
│  ├─ loader.py                 # Carga JSON ➜ pandas.DataFrame
│  ├─ metrics.py                # Cálculo de márgenes y tiempos
│  └─ plots.py                  # Visualizaciones (matplotlib)
├─ tests/
│  └─ test_metrics.py           # Unit‑tests básicos
├─ requirements.txt             # Dependencias Python
└─ README.md                    # Este documento
```

## Instalación rápida

```bash
python -m venv .venv && source .venv/bin/activate  # opcional, buen hábito
pip install -r requirements.txt
```

> **requirements.txt (mínimo)**
>
> ```
> pandas>=2.2
> matplotlib>=3.9
> seaborn>=0.13
> typer[all]>=0.9
> ```

## Uso esencial

```bash
python -m relay_analysis.compute_metrics \
       --input data/processed/optimized_relay_values_scenario_baseGA.json \
       --output reports/metrics_base.csv
```

Esto genera un CSV con los márgenes MT por par y un resumen del TMT del escenario.

Para obtener gráficos de dispersión Tiempo‑Principal vs. Tiempo‑Backup:

```bash
python -m relay_analysis.plots scatter \
       --input reports/metrics_base.csv \
       --savefig figures/mt_scatter_base.png
```

## Notebook interactivo

En `notebooks/analysis_pairs.ipynb` encontrarás un ejemplo paso a paso para:

1. Cargar varios escenarios.
2. Comparar ajustes iniciales vs. optimizados.
3. Visualizar la convergencia del GA.

## Licencia

MIT © 2025 Gustavo Arteaga
