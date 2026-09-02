# Results

Run `scripts/evaluate.py` after training to create a model-specific folder with:

- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.png`
- `roc_curves.png`

Final cross-model metrics should be compared only after all architectures have
been trained with the same patient-wise folds.

The MATLAB filter-selection analysis writes the following files to
`results/filter_selection/`:

- `filter_metrics_per_image.csv`
- `filter_metrics_summary.csv`
- `filter_pair_ranking.csv`
