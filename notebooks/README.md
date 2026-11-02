# Kaggle experiment notebook

The updated augmentation/stability notebook is
[efficientvit_mhla_kaggle_v2.ipynb](efficientvit_mhla_kaggle_v2.ipynb).
It expects a matching common ImageNet checkpoint (or explicit scratch opt-in)
and uses a fresh `mhla_v2_outputs` folder. Keep the previous notebook/results
for the original pilot. The recipe and experimental limits are documented in
[TRAINING_RECIPE.md](../applications/efficientvit_seg/TRAINING_RECIPE.md).

Regenerate v2 without replacing the original notebook:

```text
python scripts/build_kaggle_notebook.py --output notebooks/efficientvit_mhla_kaggle_v2.ipynb
```

Upload [efficientvit_mhla_kaggle.ipynb](efficientvit_mhla_kaggle.ipynb) to Kaggle.
Enable a GPU and Internet, then attach Cityscapes train/validation images and
original `gtFine_labelIds` masks. Run the cells in order.

The notebook embeds a snapshot of this project's experiment code, so no separate
repository upload is needed. It includes E0–E3 training, checkpoint resume,
full-resolution validation, per-class IoU, architecture profiling and CSV/plot
exports. Set `RUN_GROUP_ABLATION=True` for E2 M=1,2,4,8, and use multiple `SEEDS`
for final comparisons.

The default is the full 12,000-step comparison, one seed, one GPU, with matched
batch/crop/optimizer settings. It may take several sessions. Use `SMOKE_TEST=True`
to check the pipeline first; smoke scores are not research results.

Save notebook outputs before ending a session. To continue, attach the saved
output and set `PREVIOUS_OUTPUT` to its `mhla_outputs` folder. Keep the protocol
unchanged. Resume restores training state but does not reproduce the exact RNG
state or data-loader position of an uninterrupted run.

To refresh the embedded source after project edits, run from the project root:

```text
python scripts/build_kaggle_notebook.py
```

Dataset details and all editable run settings are documented inside the notebook.
Kaggle settings and output saving are described in the
[official notebook documentation](https://www.kaggle.com/docs/notebooks).
