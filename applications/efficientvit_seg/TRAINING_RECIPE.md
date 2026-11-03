# Cityscapes training recipe v2

This recipe tests EfficientViT E0--E3 under a shared, affordable protocol. It
does not reconstruct the unreleased EfficientViT segmentation training setup
or promise the published accuracy. Keep existing scratch results as pilot runs.

## What changed

| Setting | Legacy pilot | `--recipe cityscapes_v2` |
|---|---|---|
| Random scale | 0.5--2.0 times source image | Same |
| Random crop | One random location | Up to 10 locations; prefer no class above 75% of valid crop pixels |
| Horizontal flip | Probability 0.5 | Same |
| Color augmentation | None | Random brightness, contrast, saturation, hue; RGB only |
| Initialization | Scratch allowed by default | Common local pretrained checkpoint required; scratch needs explicit opt-in |
| Backbone LR | Same as decoder | 0.1 times decoder LR when using pretrained initialization |
| Optimizer/loss | AdamW / cross-entropy | Same |
| Schedule | Warmup + cosine | Same; choose total steps before starting |
| Validation masks | Resized to input dimensions | Original masks; resize logits to original dimensions |
| MHLA stabilization | Epsilon 1e-15, unconstrained mixing | Epsilon 1e-6, Q/K floor 1e-6, mixing weights clamped to [0,1] after updates |
| Diagnostics | Loss/mIoU | Also gradient norm, AMP scale, mixing statistics, sampled denominator/output statistics |

Crop retries are a preference, not a guarantee of balanced classes: after 10
attempts the last crop is used. Ignore pixels never count towards the 75% test.
All geometric transforms apply identically to RGB and masks; masks use nearest
neighbor interpolation. Validation has no random or color augmentation. Missing
masks and mismatched image/mask dimensions now raise errors instead of silently
reducing the dataset.

The augmentation choices follow common practice illustrated by
[MMSegmentation's Cityscapes pipeline](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/dsdl/cityscapes.py).
They run using the existing NumPy/OpenCV dependencies. The MHLA changes follow
the reference epsilon/QK handling and the
[authors' coefficient-clipping guidance](https://github.com/DAGroup-PKU/MHLA/issues/9).
These safeguards are a versioned experimental change, not proof of what caused
the previous collapse. The adaptation's denominator formula still needs a
separate equivalence audit against the reference operator before paper-fidelity
claims are made. Use the same stabilization in both E2 and E3.

## Order of experiments

1. Evaluate an official E0 checkpoint on all 500 Cityscapes validation images
   using the original evaluator at 1024x2048. Compare with the same model/split
   in the published benchmark. This validates evaluation independently of training.
2. Run a 100-step E0 smoke test with v2, then a 3,000-step E0 pilot. Inspect
   paired image/mask samples, class IoUs, and loss/validation curves. Tune the
   shared recipe on E0 before freezing it for the architecture comparison.
3. Run E2 past the previous 6,000--8,000-step failure point, checking sampled
   denominator minima, negative denominators, activations, AMP scale and gradients.
   A short smoke test cannot establish long-run stability. Compare E1 and E2 M=1
   numerically with matched weights; separate training runs need not match exactly.
4. Train E0--E3 with the frozen recipe, the same common initialization source,
   crop, effective batch, number of updates, evaluation, and seeds. Report
   both best and final validation scores, convergence curves and any failures.
5. Repeat final comparisons across seeds, ideally three. Select checkpoints
   consistently, and report final full-resolution results as validation results.

The E1/E2 projection is narrower after removing multi-scale QKV. Those models
also have fewer parameters and incompatible projection tensors. Report this
capacity and initialization difference; the experiment measures the practical
removal/replacement, not a perfectly capacity-matched attention comparison.

## Direct training on one GPU

Use an ImageNet classification checkpoint for the matching EfficientViT B1
backbone as the common initialization. It loads backbone tensors and leaves the
segmentation decoder initialized. The initialization report lists all loaded,
missing, unexpected and shape-incompatible tensors. A segmentation checkpoint
instead defines a fine-tuning experiment and must be labeled accordingly.

```bash
python applications/efficientvit_seg/train_efficientvit_seg_model.py \
  --recipe cityscapes_v2 \
  --data-root /datasets/cityscapes \
  --model efficientvit-seg-b1-cityscapes \
  --init-checkpoint /checkpoints/efficientvit_b1_r224.pt \
  --experiment e0 --output-dir runs/v2/e0 \
  --batch-size 2 --grad-accum-steps 2 \
  --max-steps 12000 --warmup-steps 500 \
  --crop-height 512 --crop-width 1024 \
  --val-height 512 --val-width 1024 --val-max-samples 100 \
  --resume
```

This is still a budgeted pilot: global batch 4 for 12,000 steps gives 48,000
sample presentations, about 16 dataset-equivalent passes over 2,975 images.
Gradient accumulation does not increase BatchNorm's per-forward batch size.
If batch-2 BN is unstable, investigate its statistics before changing the
normalization policy for every model. On multi-GPU CUDA, SyncBN is enabled.

For a genuine scratch ablation, add `--allow-scratch` and omit the initialization
checkpoint; the backbone LR multiplier then defaults to 1.0. For a pipeline
smoke test, use 100 steps, warmup 10, crop 256x512, and a small validation subset.
Do not infer final accuracy from that run.

## SLURM

```bash
sbatch scripts/slurm/run_mhla_experiments.sbatch \
  --mode train-eval-benchmark --recipe cityscapes_v2 \
  --experiments e0,e1,e2,e3 --groups 4 \
  --train-data /datasets/cityscapes \
  --eval-path /datasets/cityscapes/leftImg8bit/val \
  --init-checkpoint /checkpoints/efficientvit_b1_r224.pt \
  --output-root /scratch/$USER/mhla_v2 \
  --max-steps 12000 --val-max-samples 100
```

Eight GPUs per variant with batch 2 gives global batch 16, so this presents
four times as many samples as the single-GPU batch-4 command above. These are
different protocols. To compare hardware configurations, explicitly match
global batch and sample exposure. Keep the six-hour cutoff and resume as needed;
completion time must be measured on your hardware.

The 100-image subset is for frequent checkpoint selection. The subsequent
original evaluator uses the full validation directory at full resolution.
For final experiments, use all 500 images for selection when affordable.

## Reproducibility and outputs

Use a fresh output root for v2. Resume refuses changed recipes, schedules,
initialization paths, batches, or other training settings. Resume restores the
model, optimizer and scaler, but not the exact RNG state or loader position;
record interruptions. Do not extend `max-steps` mid-run because that changes
the cosine schedule. Plan the longer schedule before starting a new experiment.

Training and validation loaders have separate seeded generators. This avoids
architecture construction consuming the RNG used for worker augmentation seeds.
`metrics.jsonl` includes sampled MHLA denominator/output statistics per module
every logging interval, plus gradient norm, AMP scale and mixing-weight ranges.
Diagnostics are observations, not a guarantee that every intervening step is stable.
Non-finite loss/logits fail explicitly. AMP may skip updates with non-finite
gradients; inspect the gradient norm and scaler history if progress stalls.

Best checkpoints retain evaluator-compatible weights; `last.pt` retains restart
state. Native-mask v2 validation differs from legacy resized-mask validation:
compare all variants under the same v2 metric. A full-resolution reporting pass
can differ from lower-resolution checkpoint-selection validation and should be
labeled with its input resolution.

To isolate causes, first compare E0 legacy vs E0 v2, then ablate photometric
augmentation (`--no-photometric`) or crop retries (`--cat-max-ratio 1.0`) one at
a time if needed. Do not tune different augmentation recipes for each variant.
Additional boundary losses, distillation, auxiliary heads and decoder redesigns
would introduce further variables and are deferred.
