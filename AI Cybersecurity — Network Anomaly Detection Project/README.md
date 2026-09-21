# Network anomaly detection dashboard

A multi-page Streamlit app for exploring three unsupervised anomaly detectors (Autoencoder, Isolation Forest,
One-Class SVM) trained on normal traffic from the CICIDS2017 dataset.

| Page | What it does |
|---|---|
| **Home** | Project overview, pipeline, model cards, results recorded in the notebook |
| **📊 Interactive Dashboard** | Pick a model; KPI cards, score trend with threshold, traffic donut, top anomalies, score distribution, filterable per-flow table with CSV export |
| **📈 Model Comparison** | Metrics table with an "always flag" baseline, confusion matrices, ROC and PR curves on one plot, auto-generated analysis. Evaluates on the **recreated held-out test split** |
| **📁 Data Explorer** | Cleaning report, feature statistics, filterable table, class-coloured histograms, correlation heatmap |
| **⚙️ Settings and info** | Files in `models/`, thresholds, hyperparameters, environment, explanations of the models |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put cicids2017_binary_balanced.csv in data/   (see data/README.md)
streamlit run Home.py
```

Without the CSV the app still starts; upload any compatible file from the sidebar.

**Before you publish:** set `AUTHOR`, `GITHUB_URL` (and optionally `PORTFOLIO_URL`, `REPO_URL`) in
`utils/config.py`. They are placeholders.

## Things this project found in the original files

These are fixed or handled in the app, and worth knowing about:

1. **`isolation_forest_threshold.pkl` held the SVM's threshold (-16.417).** In the notebook, cell 84 saves the
   variable `best_t` for both models, but by then `best_t` had been overwritten by the SVM cell. The Isolation
   Forest's real threshold (printed by the notebook) is **-0.143621**. The `models/` folder in this project
   contains a corrected file. The app also validates every saved threshold against the range its model can
   produce, so the old file would be caught and replaced with the notebook value rather than silently used.
   To fix the notebook, save separate variables (for example `iso_best_t` and `svm_best_t`).
2. **The Autoencoder's saved threshold (0.000201) flags every flow as an attack**, although its ROC-AUC (0.90)
   is the best of the three models. The threshold was chosen by maximising F1 on a split that is about
   two-thirds attack, where "flag everything" already scores well. The dashboard warns when a threshold
   degenerates and offers *Balanced (Youden's J)* and *Top %* modes; the comparison page adds a
   flag-everything baseline row so this can't hide inside an F1-score.
3. **`requirements.txt` pinned scikit-learn `<1.7`, but the pickles were created with 1.7.2.** The pin is now
   `>=1.7.2,<1.8`. Streamlit is `>=1.50` because the app uses `width="stretch"` and `st.segmented_control`.
4. **The original `app.py` loaded `threshold.pkl`, which does not exist.** Each model now loads its own file.
5. The full CSV is bigger than Streamlit's 200 MB upload limit, so the app opens the file from `data/` by default
   and `.streamlit/config.toml` raises `maxUploadSize` to 1024 MB.

## Design decisions

- **Honest evaluation.** `utils/data.py::notebook_split` recreates the notebook's exact train / threshold / test
  partition (same `random_state=42` recipe, applied to index arrays). The comparison page defaults to the test
  split, and Balanced thresholds are fitted on the separate threshold split, so test metrics are not leaked into
  the cut-off. It reports whether the recreated split sizes match the notebook's.
- **Plotly only.** All charts, including heatmaps and class-coloured histograms, use Plotly, so no seaborn or
  matplotlib dependency is needed and every chart is interactive.
- **Graceful failure.** A missing or unreadable model file disables only that model; a missing dataset, wrong
  columns, empty or unparsable CSV produce a plain-language message instead of a traceback.
- **Caching.** Models load once (`st.cache_resource`) and reload if a file in `models/` changes; scores are
  cached on the data they were computed from.

## Layout

```
Home.py                         entry point
pages/                          the four pages
utils/
  config.py                     paths, model registry, palette, notebook reference values, YOUR NAME/LINKS
  artifacts.py                  loading + threshold validation (no Streamlit imports)
  data.py                       CSV reading, cleaning, notebook split (no Streamlit imports)
  scoring.py, metrics.py        scoring, thresholds, metrics, generated analysis text
  plots.py                      Plotly figures
  ui.py, services.py            theme/CSS, KPI cards, caching and the shared dataset selector
models/                         serialized scaler, models and thresholds
data/                           put the CSV here (git-ignored)
.streamlit/config.toml          theme and upload limit
```

## Deploying

Streamlit Community Cloud works with this layout, but the CSV is not in the repository (about 350 MB).
Either host the file somewhere and point `ANOMALY_DATASET_PATH` at it, or let visitors upload a sample.
TensorFlow makes the environment large; the free tier's memory is tight for the full 850k-row file, so consider
a smaller default sample or a container host.

## How this was tested

Everything ran under the pinned versions (Streamlit 1.64, TensorFlow 2.21, scikit-learn 1.7.2, pandas 2.3).
The model files load and score; the split reconstruction matches all five partition sizes printed in the
notebook and selects the same rows as the notebook's code; every page runs without errors across all model and
threshold combinations; failure paths (no data, missing models, corrupt or wrong-format CSVs) were exercised;
and the charts were rendered to images and inspected.

Not verified: the real CICIDS2017 CSV was not available, so testing used a synthetic file generated through your
scaler. The numbers on the comparison page should be checked against the notebook once you add the real data:
on the held-out test split, full set and saved thresholds, the page includes a "Compare with the notebook's
recorded results" table that reports the difference. The app was not opened in a real browser; layout and
typography were checked through rendered charts and the page structure, not by eye in Streamlit.
