# Running E1–E3 on TC2 (one arm per account)

Each TC2 account gets 1 GPU and 6-hour jobs. One arm of 40,000 steps takes about 5.7 h, so one
account runs one arm with a 2-job chain. The recipe is fixed in `scripts/protocol.env`; do not edit it.

| Arm | Attention | Multi-scale | Who |
|-----|-----------|-------------|-----|
| E0  | LiteMLA   | yes         | kieu0003 (running) |
| E1  | LiteMLA   | no          | teammate A |
| E2  | MHLA (4 groups) | no    | teammate B |
| E3  | MHLA (4 groups) | yes   | teammate C |
| E4/E5 | MALA    | —           | not implemented yet |

## 0. Connect
- VPN: GlobalProtect `vpngate-student.ntu.edu.sg` (Linux: `gpclient --browser manual connect vpngate-student.ntu.edu.sg`).
- `ssh <your_username>@10.96.189.12`

## 1. Get the code (branch `ryan/tc2-training`; after the PR is merged, use `codex/mhla-efficientvit` from trigersole)
```bash
git clone -b ryan/tc2-training https://github.com/kieusontungwork-oss/EfficientnetVit-MHLA.git ~/EfficientnetVit-MHLA
cd ~/EfficientnetVit-MHLA
```

## 2. Set up env + checkpoints (~10 min, run once)
```bash
bash scripts/tc2/setup_account.sh        # must end with "SETUP_DONE" and two "OK" sha lines
```

## 3. Copy Cityscapes into your own home (accounts cannot read each other's folders)
From your laptop (the extracted Kaggle data, `leftImg8bit/` + `gtFine/`, ~7.7 GB, ~20 min over VPN):
```bash
rsync -a --partial --info=progress2 /path/to/cityscapes/ <your_username>@10.96.189.12:datasets/cityscapes/
```
Then on TC2:
```bash
~/envs/effvit/bin/python scripts/tc2/check_cityscapes.py ~/datasets/cityscapes   # must print DATASET_OK
```

## 4. Optional sanity check: R0 (1 min of GPU)
```bash
sbatch scripts/slurm/tc2_r0_eval.sbatch
# logs/effvit-r0-<job>.out must end with: mIoU = 80.547
```

## 5. Launch your arm
```bash
bash scripts/slurm/tc2_submit_chain.sh e1 2     # or e2 / e3 — exactly your assigned arm
squeue -u $USER
```
- Job 1 trains ~5.9 h, saves `last.pt` 5 min before the limit; job 2 resumes, finishes, then runs the
  official evaluator.
- Progress: `tail -f logs/train-e1-<job>.out` or `~/runs/protocol_v1/e1_m4_s0/metrics.jsonl`.
- If both jobs end and `final.pt` is still missing, run `bash scripts/slurm/tc2_submit_chain.sh e1 1` again
  (it resumes from `last.pt`).

## 6. Report back
Send these files from `~/runs/protocol_v1/<arm>_m4_s0/`:
- `eval_full.txt` (the reported number: official evaluator, 500 val images, 1024×2048, `final.pt`)
- `metrics.jsonl`, `config.json`, `initialization_report.json`
- the first 3 lines of every `logs/train-<arm>-*.out` (job id, node, git revision)

Expected init report: E1 `loaded=247 sliced=7`, E2 `loaded=247 sliced=7` + 7 fresh MHLA weights,
E3 `loaded=261` + 7 fresh MHLA weights. Anything else: stop and tell Ryan.
