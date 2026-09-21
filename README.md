# Data

Place `cicids2017_binary_balanced.csv` in this folder. The Interactive Dashboard, Model Comparison and
Data Explorer pages open with it by default.

The file is not committed (see `.gitignore`) because it is about 350 MB. To keep it elsewhere, set
`ANOMALY_DATASET_PATH=/path/to/cicids2017_binary_balanced.csv` before starting the app.

Expected columns: the 51 CICIDS2017 flow features used in the notebook (plus `Subflow Fwd Bytes`, which is
dropped on load) and `Attack_Binary` (0 = normal, 1 = attack).
