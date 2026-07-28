# Leveraging Clustering for Large-Scale Time-Series Forecasting

This repository contains the work of an MSc team project completed for
*Recent Developments in Knowledge Discovery and Databases* at the University
of Vienna. The project investigates whether grouping households with similar
electricity-consumption patterns can support large-scale time-series
forecasting. Household histories from 2023 are analysed and clustered before
the resulting group structure is used in the forecasting stage for 2024.

## Individual contribution

This repository is a fork of the
[original team repository](https://github.com/hemalii3/rdkd). My individual
contribution was the exploratory analysis, preprocessing, feature construction,
and clustering stage. This work is documented in the first three notebooks:

| Notebook | My work |
| --- | --- |
| [`01_EDA.ipynb`](notebooks/01_EDA.ipynb) | Examined data quality, household-level consumption distributions, temporal patterns, heterogeneity, and special consumption regimes such as all-zero, near-constant, and zero-heavy series. |
| [`02_Preprocessing_Features.ipynb`](notebooks/02_Preprocessing_Features.ipynb) | Developed the cleaning and feature-preparation workflow, including the treatment of degenerate series, construction of behavioural time-series features, scaling, feature filtering, and preparation of method-specific clustering inputs. |
| [`03_Clustering_Experiments.ipynb`](notebooks/03_Clustering_Experiments.ipynb) | Designed and evaluated the clustering experiments, comparing feature-based Ward hierarchical clustering with k-Shape across candidate cluster counts and assessing cluster quality, stability, balance, and interpretability. |

The associated analysis uses the reusable preprocessing and clustering
components under [`src/preprocessing`](src/preprocessing) and
[`src/clustering`](src/clustering). It produces household-level cluster labels,
diagnostic summaries, cluster profiles, and cached results for the downstream
forecasting pipeline.

## Clustering workflow

The clustering stage follows four main steps:

1. **Exploratory analysis and validation**  
   Inspect household series for missing or negative values, zeros, extreme
   consumption, temporal structure, and heterogeneous behavioural patterns.

2. **Preprocessing and feature construction**  
   Separate degenerate consumption regimes from the main clustering fit while
   retaining genuine consumption spikes. Construct interpretable feature
   families describing weekly and seasonal behaviour, variability, persistence,
   trend, energy profiles, and normalized time-series shape.

3. **Alternative clustering representations**  
   Compare:

   - **Feature-based Ward clustering**, applied to selected and scaled
     behavioural features; and
   - **k-Shape**, used as a sequence-shape benchmark on row-wise normalized
     household time series.

4. **Evaluation and interpretation**  
   Examine candidate cluster counts using internal quality measures, cluster
   balance and size, resampling-based stability, representative series, and
   weekly, monthly, and normalized cluster profiles. The selected labels are
   then exported for the forecasting stage.

## Repository structure

```text
.
├── Data/                 # 2023 and 2024 household-consumption data
├── notebooks/            # Ordered analysis and forecasting notebooks
├── src/
│   ├── clustering/       # Clustering, evaluation, and reporting utilities
│   ├── forecasting/      # Downstream forecasting pipeline
│   ├── preprocessing/    # Cleaning, EDA, feature engineering, and selection
│   └── utils/            # Configuration and data-loading utilities
├── results/              # Cached clustering outputs and forecasting results
└── requirements.txt
```

## Running the project

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Run the notebooks in numerical order. The first three notebooks reproduce the
clustering stage; notebooks 04–06 continue with forecasting and evaluation.
Project paths and experiment settings are defined in
[`src/utils/config.py`](src/utils/config.py), and generated artifacts are stored
under `results/`.

## Main technologies

Python, pandas, NumPy, SciPy, scikit-learn, tslearn, Matplotlib, seaborn,
LightGBM, and Jupyter.
