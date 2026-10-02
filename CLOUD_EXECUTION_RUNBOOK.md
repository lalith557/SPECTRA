# SPECTRA — Cloud Execution Runbook

Ordered procedure to run the frozen experiment protocol on a cloud GPU and freeze the evidence.
**Do not resume the throttled laptop run. P1 starts from scratch. Every number records full provenance.**

## STEP 1 — Provision GPU
One GPU, ≥8 GB VRAM (24 GB recommended): L4 / A10 / RTX 4090 / A5000 / A100. CUDA 12.8 driver.

## STEP 2 — Clone exact repo state
```bash
git clone <repo-url> spectra && cd spectra
git checkout 5ca9f0f
```
Apply the **uncommitted corrected-RAFT change** to `flow/raft_wrapper.py` (`_denorm` returns `(x*2.0-1.0).clamp(-1,1)`), and ensure `configs/config.yaml` has `wandb_project: null`. See `CLOUD_GPU_SETUP.md` §9.

## STEP 3 — Install pinned environment
```bash
python -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
pip install -r requirements-cloud.txt
```

## STEP 4 — Preflight
```bash
python scripts/cloud_preflight.py --load-models      # must print PREFLIGHT PASSED
```

## STEP 5 — RAFT sanity test
```bash
python scripts/raft_sanity_test.py                   # must print RAFT SANITY PASSED
```

## STEP 6 — Verify dataset counts
Covered by preflight (Trans10K 5000/1000/4428; VGSD 105 clips; ClearPose scenes). Re-check if transferring datasets separately.

## STEP 7 — Verify checkpoint / pretrained-weight availability
P1/P2/P3 need **no** input checkpoint (train from scratch). Confirm DINOv2 (torch.hub) + RAFT (torchvision) weights download or are cached (preflight `--load-models` covers this).

## STEP 8 — Record provenance (before P1)
Into the experiment dir's `ENVIRONMENT.txt` / `GIT_STATE.txt`:
```bash
git rev-parse HEAD; git status --porcelain; git diff > GIT_STATE.diff
python -c "import torch,torchvision,sys;print(sys.version);print(torch.__version__,torchvision.__version__,torch.version.cuda);print(torch.cuda.get_device_name(0))"
pip freeze > ENVIRONMENT.txt
date -u +%Y-%m-%dT%H:%M:%SZ
```

## STEP 9 — Authorize and run (manual gate per experiment)
Only after STEP 4–8 pass, and only with explicit user authorization per experiment:
1. **P1** (mandatory): `python scripts/run_corrected_headline.py`  → corrected headline SPECTRA.
2. **P1.5** (inference): `python eval/local_cross_dataset_eval.py --checkpoint EXPERIMENTS/13_corrected_headline_2026-10-02/checkpoints/spectra_best.pth --max 1500 --out EXPERIMENTS/13_corrected_headline_2026-10-02/crossdataset_corrected.json`
3. **P2** (strongly recommended): corrected Trans10K OFCV ablation (ofcv_only, ofcv_brf), 30 ep — fresh dir `EXPERIMENTS/13_corrected_trans10k_ablation/`.
4. **P3** (recommended): `python scripts/run_vgsd_corrected.py` → VGSD OFCV variants, 336/12ep.
5. **P4** (optional): multi-seed, see `CLOUD_GPU_SETUP.md` §18.

**P1 must start from scratch. Never resume the thermally throttled laptop run (`EXPERIMENTS/13_...` was produced locally under throttling; a cloud re-run is a separate fresh dir if desired).**

---

## Result directory design (applies to every run)
```
EXPERIMENTS/
  12_corrected_final/            # (reserved) corrected headline, cloud clean-timing re-run if done
  13_corrected_headline_2026-10-02/   # P1 completed locally under throttling (valid; see its README)
  13_corrected_trans10k_ablation/     # P2
  11_corrected_raft_2026-10-02/vgsd_4way/   # P3 (driver target) — or a fresh 14_ dir
  15_multiseed_optional/              # P4
```
Each experiment directory MUST contain:
- `GIT_STATE.txt` (HEAD + full diff) · `CONFIG.yaml` (snapshot) · `COMMAND.txt` (exact command) · `ENVIRONMENT.txt` (pip freeze + CUDA/GPU) · `run.log` (raw) · per-epoch metrics (JSON) · `checkpoints/` · final `result.json` / `meta.json` · `README.md`.
**Never overwrite an earlier experiment. Never overwrite `checkpoints/spectra_best.pth`.**

## Terminate / restart safely
- Kill = Ctrl-C / stop the process. A killed run leaves per-epoch checkpoints; treat a partial dir as non-final and start a fresh dir for the clean run.
- Do not mix protocols across a resume. Prefer fresh from-scratch for publication.
