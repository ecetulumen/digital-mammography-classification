# Exploratory notebooks

These cleaned notebooks preserve the original Google Colab experiments from
the graduation project. Repeated example cells and long epoch-by-epoch logs
were removed for readability.

The canonical, reusable implementation is under `src/` and `scripts/`. Use that
pipeline for new experiments because it applies augmentation only to training
data and keeps patients separated across train, validation, and test folds.

