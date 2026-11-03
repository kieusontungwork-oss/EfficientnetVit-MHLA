"""Build a portable notebook containing the current experiment source snapshot."""

import ast
import argparse
import base64
import hashlib
import io
import json
import subprocess
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
builder_args = argparse.ArgumentParser()
builder_args.add_argument("--output", type=Path, default=ROOT / "notebooks" / "efficientvit_mhla_kaggle.ipynb")
OUT = builder_args.parse_args().output
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})


def code(source):
    source = textwrap.dedent(source).strip() + "\n"
    ast.parse(source)
    cells.append({"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None})


md("""
# EfficientViT + MHLA: complete Cityscapes experiment

Runs the project's E0–E3 comparison, optional E2 group-count ablations,
training, best-checkpoint evaluation, profiling, plots and CSV exports.
The current project code is embedded below; no GitHub clone or code dataset is required.

**Before running:** upload this `.ipynb` to Kaggle, enable a **GPU** and **Internet**,
and attach Cityscapes with original RGB images and **`*_gtFine_labelIds.png`** masks.
Color masks or `labelTrainIds` alone are not supported. Keep the original folder layout:

```
cityscapes/
  leftImg8bit/train/<city>/*_leftImg8bit.png
  leftImg8bit/val/<city>/*_leftImg8bit.png
  gtFine/train/<city>/*_gtFine_labelIds.png
  gtFine/val/<city>/*_gtFine_labelIds.png
```

Images and labels may be in separate attached datasets; discovery links them into one root.
Extract ZIP datasets first if necessary. Set `DATA_ROOT` below if automatic discovery is ambiguous.
Only attach data you are permitted to use; this notebook does not download Cityscapes.

Run cells in order. The default is the full 12,000-step protocol for four variants,
one seed, sequentially on one GPU. It can require **multiple Kaggle sessions**.
Set `SMOKE_TEST=True` for a short pipeline check whose scores are not final results.
For final conclusions use multiple seeds and enable the group-count ablation.
The supplied trainer is a controlled comparison protocol, not the paper's unreleased recipe.
This v2 notebook adds crop retries, RGB photometric augmentation, native-mask
validation, pretrained-backbone LR scaling and MHLA stability safeguards.
Attach one matching ImageNet classification checkpoint for every model, or set
`ALLOW_SCRATCH=True` explicitly for a scratch ablation. Use a fresh v2 output folder.
""")

code('''
from pathlib import Path
import os, sys, json, time, subprocess, shutil, hashlib, platform

DATA_ROOT = None  # e.g. "/kaggle/input/my-cityscapes/cityscapes"
IMAGE_ROOT = None  # optional explicit folder containing train/ and val/ image folders
LABEL_ROOT = None  # optional explicit folder containing train/ and val/ mask folders
PREVIOUS_OUTPUT = None  # e.g. "/kaggle/input/previous-notebook/mhla_outputs"
INIT_CHECKPOINT = None  # common trusted local .pt for ALL variants, or None = scratch
RECIPE = "cityscapes_v2"
ALLOW_SCRATCH = False  # explicitly opt in for a scratch ablation; otherwise attach common ImageNet weights

SMOKE_TEST = False
SEEDS = [0]  # use e.g. [0, 1, 2] for final comparisons
VARIANTS = ["e0", "e1", "e2", "e3"]
RUN_GROUP_ABLATION = False  # adds E2 M=1,2,8; M=4 reuses the main E2 run
GROUP_COUNTS = [1, 2, 4, 8]
MODEL = "efficientvit-seg-b1-cityscapes"
MAX_STEPS = 8 if SMOKE_TEST else 12000
CROP = (128, 256) if SMOKE_TEST else (512, 1024)
EVAL_SIZE = (128, 256) if SMOKE_TEST else (1024, 2048)
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2  # effective batch = 4, one GPU
WORKERS = 2
LR = 6e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 2 if SMOKE_TEST else 500
SAVE_INTERVAL = 4 if SMOKE_TEST else 250
EVAL_INTERVAL = 4 if SMOKE_TEST else 1000
VAL_MAX_SAMPLES = 4 if SMOKE_TEST else 0  # 0 = full validation split
TRAINING_BUDGET_HOURS = 8.0  # training budget; leave time for evaluation and output saving
BENCH_WARMUP = 2 if SMOKE_TEST else 50
BENCH_ITERATIONS = 5 if SMOKE_TEST else 200
RUN_TESTS = True

WORK = Path("/kaggle/working")
REPO = WORK / "efficientvit_mhla_source"
OUTPUT = WORK / ("mhla_v2_smoke_outputs" if SMOKE_TEST else "mhla_v2_outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)
assert MODEL.endswith("-cityscapes"), "This notebook's data setup targets Cityscapes."
assert SEEDS and len(SEEDS) == len(set(SEEDS))
assert VARIANTS and set(VARIANTS) <= {"e0", "e1", "e2", "e3"}
assert all(m > 0 for m in GROUP_COUNTS)
assert TRAINING_BUDGET_HOURS > 0
assert RECIPE == "cityscapes_v2", "Use the original notebook for the legacy protocol."
assert INIT_CHECKPOINT or ALLOW_SCRATCH or SMOKE_TEST, "Attach a common ImageNet checkpoint or explicitly set ALLOW_SCRATCH=True."
print("Mode:", "SMOKE CHECK" if SMOKE_TEST else "FULL EXPERIMENT")
print("Effective training batch:", BATCH_SIZE * GRAD_ACCUM_STEPS)
''')

# Include only tracked source/config/docs. Read current working files to preserve edits.
paths = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
paths = sorted(set(paths) | {"applications/efficientvit_seg/TRAINING_RECIPE.md"})
bundle = io.BytesIO()
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for relative in sorted(paths):
        p = ROOT / relative
        if p.is_file() and (relative.startswith("efficientvit/") and p.suffix == ".py"
                            or relative.startswith("applications/efficientvit_seg/") and p.suffix in {".py", ".yaml", ".md"}
                            or relative.startswith("tests/") and p.suffix == ".py"
                            or relative in {"LICENSE", "MHLA_EXPERIMENTS.md"}):
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
raw = bundle.getvalue()
digest = hashlib.sha256(raw).hexdigest()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
md("""
## Restore the bundled experiment and install dependencies

Keeps Kaggle's existing PyTorch/CUDA stack. Only missing supporting packages are installed.
Internet is needed for missing packages; GPU availability is checked before training.
""")
code(f'''
import base64, io, zipfile, importlib.util
SOURCE_SHA256 = {digest!r}
SOURCE_COMMIT = {commit!r}
SOURCE_B64 = {base64.b64encode(raw).decode()!r}
source_bytes = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256
REPO.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
    for name in archive.namelist():
        target = (REPO / name).resolve()
        assert target.is_relative_to(REPO.resolve()), name
    archive.extractall(REPO)
sys.path.insert(0, str(REPO))
os.chdir(REPO)
packages = {{"timm": "timm", "torchprofile": "torchprofile", "onnx": "onnx",
            "onnxsim": "onnxsim", "yaml": "pyyaml",
            "einops": "einops", "cv2": "opencv-python-headless", "tqdm": "tqdm",
            "matplotlib": "matplotlib", "pandas": "pandas"}}
missing = [pip_name for module, pip_name in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
import torch, numpy as np, pandas as pd
assert torch.cuda.is_available(), "Enable a GPU in Kaggle Session options, then restart."
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Using one GPU per experiment; variants run sequentially.")
print("Embedded source SHA256:", SOURCE_SHA256)
''')

md("""
## Find and validate the dataset

Checks every train/validation image has its original ID mask, inspects labels, and
saves counts. Full mode requires the official 2,975 training and 500 validation pairs.
RGB label masks and train-ID-only datasets produce an error before training.
""")
code('''
from PIL import Image
from applications.efficientvit_seg.train_efficientvit_seg_model import SegmentationDataset

def find_root(folder_name):
    candidates = sorted({p for p in Path("/kaggle/input").rglob(folder_name)
                         if p.is_dir() and (p / "train").is_dir() and (p / "val").is_dir()})
    if len(candidates) != 1:
        raise RuntimeError(f"Found {len(candidates)} {folder_name} roots: {candidates}. "
                           "Set DATA_ROOT or both IMAGE_ROOT and LABEL_ROOT explicitly.")
    return candidates[0]

if DATA_ROOT:
    DATA_ROOT = Path(DATA_ROOT)
else:
    image_root = Path(IMAGE_ROOT) if IMAGE_ROOT else find_root("leftImg8bit")
    label_root = Path(LABEL_ROOT) if LABEL_ROOT else find_root("gtFine")
    DATA_ROOT = WORK / "cityscapes_links"
    DATA_ROOT.mkdir(exist_ok=True)
    for name, target in [("leftImg8bit", image_root), ("gtFine", label_root)]:
        link = DATA_ROOT / name
        if link.is_symlink():
            assert link.resolve() == target.resolve(), f"Restart or correct stale link: {link}"
        elif not link.exists():
            link.symlink_to(target, target_is_directory=True)
        else:
            raise RuntimeError(f"Existing non-link at {link}; choose DATA_ROOT explicitly.")

data_counts = {}
for split, expected in [("train", 2975), ("val", 500)]:
    dataset = SegmentationDataset(str(DATA_ROOT), "cityscapes", split, CROP, training=False)
    image_count = len(list((DATA_ROOT / "leftImg8bit" / split).glob("*/*_leftImg8bit.png")))
    assert len(dataset) == image_count, f"Missing masks for {split}: {len(dataset)}/{image_count} paired"
    if not SMOKE_TEST:
        assert len(dataset) == expected, f"Expected official {split} split: {expected}, got {len(dataset)}"
    for image_path, label_path in dataset.samples[:10]:
        with Image.open(image_path) as image, Image.open(label_path) as mask:
            assert image.size == mask.size, f"Image/mask size mismatch: {label_path}"
            labels = np.asarray(mask)
            assert labels.ndim == 2, f"Expected integer ID mask: {label_path}"
            assert set(np.unique(labels)) <= (set(range(34)) | {255}), f"Invalid label IDs: {label_path}"
    sample = dataset[0]
    assert (sample["label"] >= 0).any(), "All pixels ignored; check mask encoding."
    data_counts[split] = len(dataset)
print("Cityscapes:", DATA_ROOT, data_counts)
''')

md("""
## Restore earlier outputs and freeze the protocol

For another session, attach a saved notebook output and point `PREVIOUS_OUTPUT` at
its `mhla_outputs` folder. Checkpoints, logs and completed runs are copied before continuing.
The protocol and source fingerprint must match. To change a protocol, use a fresh output folder.
Use only trusted checkpoint files (PyTorch resume loads optimizer state using pickle).

Resume restores model, optimizer, AMP scaler, step and schedule. The current project
trainer does **not** restore random-number states or the exact data-loader position;
resumed runs are not bitwise equivalent to uninterrupted runs. Record this in reports.
""")
code('''
if PREVIOUS_OUTPUT:
    previous = Path(PREVIOUS_OUTPUT)
    assert previous.is_dir() and (previous / "protocol.json").is_file(), "Invalid previous output folder"
    assert not any(OUTPUT.iterdir()), "Restore into a fresh output folder to avoid mixing runs."
    shutil.copytree(previous, OUTPUT, dirs_exist_ok=True)

init_hash = None
if INIT_CHECKPOINT:
    INIT_CHECKPOINT = str(Path(INIT_CHECKPOINT).resolve())
    assert Path(INIT_CHECKPOINT).is_file()
    hasher = hashlib.sha256()
    with open(INIT_CHECKPOINT, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    init_hash = hasher.hexdigest()
protocol = dict(source_sha256=SOURCE_SHA256, recipe=RECIPE, model=MODEL, max_steps=MAX_STEPS,
                crop=list(CROP), eval_size=list(EVAL_SIZE), batch_size=BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS, lr=LR, weight_decay=WEIGHT_DECAY,
                warmup_steps=WARMUP_STEPS, eval_interval=EVAL_INTERVAL,
                val_max_samples=VAL_MAX_SAMPLES, smoke=SMOKE_TEST,
                initialization_sha256=init_hash, dataset_counts=data_counts,
                amp=True, scale_range=[0.5, 2.0], grad_clip=1.0, world_size=1,
                photometric=True, cat_max_ratio=0.75, native_val_labels=True,
                backbone_lr_multiplier=0.1 if INIT_CHECKPOINT else 1.0, mhla_weight_constraint="clamp",
                mhla_eps=1e-6, mhla_qk_eps=1e-6)
protocol_path = OUTPUT / "protocol.json"
if protocol_path.exists():
    assert json.loads(protocol_path.read_text()) == protocol, "Protocol changed. Use a new output folder."
protocol_path.write_text(json.dumps(protocol, indent=2))
environment = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0), source_commit=SOURCE_COMMIT,
                   source_sha256=SOURCE_SHA256, data_root=str(DATA_ROOT),
                   resumed_from=PREVIOUS_OUTPUT, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
with (OUTPUT / "environments.jsonl").open("a") as handle:
    handle.write(json.dumps(environment) + "\\n")
with (OUTPUT / "packages.txt").open("w") as handle:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=handle, check=True)

def run_logged(command, log_path):
    print("Running:", " ".join(map(str, command)), flush=True)
    with Path(log_path).open("w") as handle:
        process = subprocess.Popen(list(map(str, command)), cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            result = process.wait()
        except BaseException:
            process.terminate()
            process.wait(timeout=180)
            raise
    if result:
        raise RuntimeError(f"Command failed ({result}); see {log_path}")

if RUN_TESTS:
    run_logged([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], OUTPUT / "tests.log")
''')

md("""
## Train E0–E3 and optional group ablations

E0 runs first for each seed. All variants share the resolved training protocol and
initialization policy. A common checkpoint is expected; scratch training requires
explicit opt-in. Compatible-load reports are retained. A timer requests a graceful
stop after the training budget, allowing a checkpoint at the end of an optimizer step.
Save outputs before the Kaggle session expires; a hard platform termination cannot be handled.
""")
code('''
import signal, threading

runs = []
for seed in SEEDS:
    for variant in VARIANTS:
        runs.append(dict(variant=variant, groups=4, seed=seed))
    if RUN_GROUP_ABLATION:
        for groups in GROUP_COUNTS:
            entry = dict(variant="e2", groups=groups, seed=seed)
            if entry not in runs:
                runs.append(entry)
for run in runs:
    run["name"] = f'{run["variant"]}_m{run["groups"]}_seed{run["seed"]}'
(OUTPUT / "run_plan.json").write_text(json.dumps(runs, indent=2))
deadline = time.monotonic() + TRAINING_BUDGET_HOURS * 3600
for run in runs:
    directory = OUTPUT / run["name"]
    directory.mkdir(exist_ok=True)
    if INIT_CHECKPOINT is None:
        (directory / "initialization_report.json").write_text(json.dumps(
            dict(policy="scratch", seed=run["seed"], common_checkpoint=None), indent=2))
    checkpoint_path = directory / "last.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        step = int(checkpoint["step"])
        del checkpoint
        if step >= MAX_STEPS:
            print("Already complete:", run["name"])
            continue
    remaining = deadline - time.monotonic()
    if remaining < 120:
        print("Training budget reached. Save outputs and resume in another session.")
        break
    command = [sys.executable, "-u", "applications/efficientvit_seg/train_efficientvit_seg_model.py",
               "--recipe", RECIPE,
               "--data-root", str(DATA_ROOT), "--dataset", "cityscapes", "--model", MODEL,
               "--experiment", run["variant"], "--mhla-num-groups", str(run["groups"]),
               "--output-dir", str(directory), "--max-steps", str(MAX_STEPS),
               "--batch-size", str(BATCH_SIZE), "--grad-accum-steps", str(GRAD_ACCUM_STEPS),
               "--workers", str(WORKERS), "--crop-height", str(CROP[0]), "--crop-width", str(CROP[1]),
               "--lr", str(LR), "--weight-decay", str(WEIGHT_DECAY), "--warmup-steps", str(WARMUP_STEPS),
               "--log-interval", str(2 if SMOKE_TEST else 20), "--save-interval", str(SAVE_INTERVAL),
               "--eval-interval", str(EVAL_INTERVAL), "--val-max-samples", str(VAL_MAX_SAMPLES),
               "--seed", str(run["seed"]), "--amp", "--no-sync-bn", "--resume"]
    if INIT_CHECKPOINT:
        command += ["--init-checkpoint", INIT_CHECKPOINT]
    if ALLOW_SCRATCH or SMOKE_TEST:
        command += ["--allow-scratch"]
    # A single trainer process owns one GPU; SIGTERM invokes its checkpoint handler.
    log = directory / f'train_{time.time_ns()}.log'
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
        def stop_training(p=process):
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        timer = threading.Timer(remaining, stop_training)
        timer.daemon = True
        timer.start()
        try:
            while process.poll() is None:
                time.sleep(20)
                lines = log.read_text(errors="replace").splitlines()
                if lines:
                    print(run["name"], lines[-1], flush=True)
        except KeyboardInterrupt:
            stop_training()
            process.wait(timeout=180)
            raise
        finally:
            timer.cancel()
    if process.returncode:
        print(log.read_text(errors="replace")[-8000:])
        raise RuntimeError(f"Training failed: {run['name']}; see {log}")
    assert checkpoint_path.exists(), "Trainer exited without a checkpoint"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed = int(checkpoint["step"]) >= MAX_STEPS
    del checkpoint
    if not completed:
        print("Saved partial run:", run["name"], "— resume from this output next session.")
        break
''')

md("""
## Evaluate best checkpoints at the reporting resolution

The best checkpoint is selected using validation images resized to the training
input size, with masks retained at their original resolution.
This cell then evaluates the complete validation split at `EVAL_SIZE`, using FP32
and the trainer's confusion-matrix metric. At the default 1024×2048 Cityscapes
resolution, labels retain their native resolution. It saves mIoU and all 19 class IoUs.
Partial training runs are recorded but excluded from the final comparison.
""")
eval_source = '''
import argparse, json, math, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from applications.efficientvit_seg.train_efficientvit_seg_model import SegmentationDataset, evaluate, extract_state_dict
from efficientvit.seg_model_zoo import create_efficientvit_seg_experiment_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--groups', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    data = SegmentationDataset(a.data_root, 'cityscapes', 'val', (a.height, a.width), False,
                               max_samples=a.max_samples, native_val_labels=True)
    loader = DataLoader(data, batch_size=1, num_workers=a.workers, pin_memory=True)
    model = create_efficientvit_seg_experiment_model(a.model, a.variant, a.groups, pretrained=False)
    model.load_state_dict(extract_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=False)), strict=True)
    model = model.cuda()
    miou, classes = evaluate(model, loader, torch.device('cuda'), 19, False, 1)
    result = dict(miou=miou, class_iou={k: v if math.isfinite(v) else None for k,v in zip(data.class_names, classes)},
                  samples=len(data), resolution=[a.height, a.width], precision='fp32', checkpoint=a.checkpoint)
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
'''
code(f'(REPO / "kaggle_evaluate.py").write_text({textwrap.dedent(eval_source)!r})')
code('''
completed_runs = []
status = []
for run in runs:
    directory = OUTPUT / run["name"]
    last = directory / "last.pt"
    steps = 0
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        steps = int(checkpoint["step"])
        del checkpoint
    status.append(dict(**run, steps=steps, complete=steps >= MAX_STEPS))
    if steps < MAX_STEPS:
        continue
    best = directory / "best.pt"
    assert best.is_file(), f"No best checkpoint for {run['name']}"
    result_path = directory / "evaluation.json"
    if not result_path.exists():
        run_logged([sys.executable, "kaggle_evaluate.py", "--data-root", DATA_ROOT,
                    "--checkpoint", best, "--model", MODEL, "--variant", run["variant"],
                    "--groups", run["groups"], "--height", EVAL_SIZE[0], "--width", EVAL_SIZE[1],
                    "--max-samples", VAL_MAX_SAMPLES, "--workers", WORKERS, "--output", result_path],
                   directory / "evaluation.log")
    completed_runs.append(run)
pd.DataFrame(status).to_csv(OUTPUT / "run_status.csv", index=False)
display(pd.DataFrame(status))
''')

md("""
## Profile parameters, estimated MACs, latency, FPS and memory

Profiles each completed architecture once at batch 1, FP32 and `EVAL_SIZE`.
The project's profiler constructs fresh weights: these measure architecture inference cost,
not prediction quality of the trained checkpoint. MAC coverage warnings are saved and MACs
must be reported as estimates. GPU model and software versions are saved with each profile.
""")
code('''
architectures = sorted({(r["variant"], r["groups"]) for r in completed_runs})
for variant, groups in architectures:
    target = OUTPUT / f"benchmark_{variant}_m{groups}.json"
    reusable = False
    if target.exists():
        old = json.loads(target.read_text())
        reusable = (old.get("gpu") == torch.cuda.get_device_name(0)
                    and old.get("torch_version") == torch.__version__
                    and old.get("input_shape") == [1, 3, *EVAL_SIZE])
    if not reusable:
        run_logged([sys.executable, "applications/efficientvit_seg/benchmark_mhla.py",
                    "--model", MODEL, "--variants", variant, "--mhla_num_groups", groups,
                    "--input_size", *EVAL_SIZE, "--batch_size", 1, "--device", "cuda",
                    "--warmup", BENCH_WARMUP, "--iterations", BENCH_ITERATIONS, "--output", target],
                   target.with_suffix(".log"))
''')

md("""
## Compare results and save figures

Final comparisons use completed runs only. Scores are percentages. Standard deviation
is meaningful only with multiple seeds. H1 compares E1 vs E0, H2 compares E2 vs E1,
and H3 compares E3 vs E2, all at M=4. Interpret differences relative to seed variation.
The class table includes small/local classes such as person, rider, pole, sign and bicycle.
""")
code('''
import matplotlib.pyplot as plt

rows, class_rows = [], []
for run in completed_runs:
    directory = OUTPUT / run["name"]
    evaluation = json.loads((directory / "evaluation.json").read_text())
    benchmark = json.loads((OUTPUT / f"benchmark_{run['variant']}_m{run['groups']}.json").read_text())
    profile = benchmark["results"][0]
    history = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
    validation = [r for r in history if r["type"] == "validation"]
    best_record = max(validation, key=lambda r: r["miou"])
    rows.append(dict(**run, miou=evaluation["miou"], parameters=profile["parameters"],
                     macs_estimate=profile["macs"], macs_notes=profile["macs_notes"],
                     latency_ms=profile["latency_ms"], fps=profile["fps"],
                     peak_gpu_memory_bytes=profile["peak_gpu_memory_bytes"],
                     best_step=best_record["step"], best_epoch=best_record["epoch"],
                     checkpoint=evaluation["checkpoint"], smoke=SMOKE_TEST))
    class_rows.append(dict(**run, **evaluation["class_iou"]))

results = pd.DataFrame(rows)
if results.empty:
    print("No runs finished yet. Save outputs and resume before making accuracy comparisons.")
else:
    results.to_csv(OUTPUT / "results.csv", index=False)
    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(OUTPUT / "per_class_iou.csv", index=False)
    summary = results.groupby(["variant", "groups"]).agg(
        seeds=("seed", "count"), miou_mean=("miou", "mean"), miou_std=("miou", "std"),
        parameters=("parameters", "first"), latency_ms=("latency_ms", "first"), fps=("fps", "first"))
    summary.to_csv(OUTPUT / "summary.csv")
    display(summary)
    display(per_class)
    main = results[results.groups == 4]
    pivot = main.pivot(index="seed", columns="variant", values="miou")
    deltas = pd.DataFrame(index=pivot.index)
    for title, left, right in [("H1_E0_minus_E1", "e0", "e1"), ("H2_E2_minus_E1", "e2", "e1"),
                                ("H3_E3_minus_E2", "e3", "e2")]:
        if left in pivot and right in pivot:
            deltas[title] = pivot[left] - pivot[right]
    deltas.to_csv(OUTPUT / "paired_seed_deltas.csv")
    display(deltas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for run in completed_runs:
        history = [json.loads(line) for line in (OUTPUT / run["name"] / "metrics.jsonl").read_text().splitlines()]
        for ax, kind, key in [(axes[0], "train", "loss"), (axes[1], "validation", "miou")]:
            records = [r for r in history if r["type"] == kind]
            ax.plot([r["step"] for r in records], [r[key] for r in records], label=run["name"])
    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel("Selection validation mIoU (%)")
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    fig.suptitle("SMOKE CHECK ONLY" if SMOKE_TEST else "EfficientViT + MHLA")
    fig.tight_layout()
    fig.savefig(OUTPUT / "learning_curves.png", dpi=180)
    plt.show()
''')

md("""
## Preserve outputs and resume

All checkpoints, configuration, source fingerprints, logs, per-class IoUs, profiles,
CSV tables and figures are in the output folder below. Save a Kaggle notebook version
**with outputs** and verify the files are present before ending the session.
Download the output folder or attach that saved notebook output as input to the next session.
Then set `PREVIOUS_OUTPUT`, keep the same protocol, and run again.

The output folder intentionally keeps `last.pt` (optimizer/scaler resume state) and
`best.pt` (evaluation weights) separately. Do not delete `last.pt` until training is done.
You can add seeds or enable group ablations in later sessions while preserving the same protocol.
Changing batch/crop/steps or initialization starts a different comparison and requires a fresh folder.

Kaggle setup and output behavior: [official notebook documentation](https://www.kaggle.com/docs/notebooks).
Project implementation notes are in the extracted `MHLA_EXPERIMENTS.md`.
""")
code('''
from IPython.display import FileLink, display
files = [p for p in OUTPUT.rglob("*") if p.is_file()]
size_gb = sum(p.stat().st_size for p in files) / 1e9
print(f"{len(files)} files, {size_gb:.2f} GB in {OUTPUT}")
print("Save this folder as notebook output before the session ends.")
for name in ["run_status.csv", "results.csv", "summary.csv", "per_class_iou.csv", "learning_curves.png"]:
    path = OUTPUT / name
    if path.exists():
        display(FileLink(str(path)))
''')

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}},
            "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell["id"] = f"mhla-{i:02d}"
    if cell["cell_type"] == "code" and "SOURCE_B64 =" in cell["source"]:
        cell["metadata"] = {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes), {len(cells)} cells, source {digest}")
