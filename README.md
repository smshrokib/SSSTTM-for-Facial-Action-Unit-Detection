# Semi-Supervised Spatio-Temporal Transformer Modeling for Facial Action Unit Detection

This repository contains research code for facial Action Unit (AU) detection on in-the-wild video data, with emphasis on:

- spatial-temporal Transformer modeling,
- audio-visual fusion,
- multi-label AU prediction,
- semi-supervised Mean Teacher training.

A local UESTC thesis LaTeX folder can be kept alongside this codebase as reference material; it is excluded from Git tracking by default.

## Repository Scope

This is a mixed workspace with two major parts:

1. AU detection research code (training, evaluation, visualization, utilities).
2. Optional local thesis writing assets in `UESTC-Thesis-Latex-Template-main/` (ignored by Git).

## Project Structure

Key folders and files:

- `models/`: model backbones and AU heads (`hrformer`, `tformer`, `avformer`, etc.)
- `dataloader/`: dataset split creation, LMDB-backed data loading, audio/clip transforms
- `metrics/`: AU/EX/VA metrics
- `teacher/`: semi-supervised consistency losses and ramp schedules
- `112_align/`: scripts for building and reading LMDB image caches
- `postprocess/`: post-processing utilities
- `train.py`, `train2.py`, `train3.py`: supervised training variants
- `train_mt.py`, `train_mt2.py`: Mean Teacher semi-supervised training variants
- `train_mtmm.py`: DDP-friendly Mean Teacher training script
- `train_all.py`: two-stage launcher (supervised stage -> MT stage)
- `test_aff2.py`, `test_aff2mm.py`: AU evaluation scripts
- `show_loss.py`, `show_result.py`, `show_CAM.py`, `show_CAMmm.py`: analysis and visualization scripts
- `opts*.py`: option presets for different experiments
- `UESTC-Thesis-Latex-Template-main/` (optional, local only): thesis source (`main.tex`) and slides template

## Environment Setup

Recommended baseline:

- Python 3.8+
- CUDA-compatible PyTorch

Install dependencies (the provided `requirements.txt` is minimal, so install core packages explicitly):

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio numpy opencv-python lmdb einops tqdm scikit-learn matplotlib pillow natsort
```

Notes:

- The original codebase was developed with older PyTorch APIs (historically around 1.6).
- If you use a newer PyTorch version, verify all training/inference scripts in your environment before long runs.

## Data Layout Expectations

The code expects a processed Aff-Wild2 style layout. Important runtime arguments:

- `--root`: dataset root containing extracted frames/audio metadata
- `--cache_dir`: cached split metadata directory (`split_dict_*.pkl`)
- `--lmdb_label_dir`: LMDB labels/images directory

Typical required components referenced by loaders:

- frame paths under `extracted/`
- `video2orignal.pkl`
- LMDBs such as:
	- `.croped_aligned_jpeg`
	- `.croped_aligned_mask` (optional)
	- `.label_au`
	- `.label_expr`
	- `.label_va`

If split caches do not exist, `dataloader/data_split.py` logic is used to generate split metadata.

## Training Workflows

### 1) Supervised training (single-stage)

Example:

```bash
python train.py \
	--task AU \
	--model_name hrformer \
	--modality A;V \
	--root /path/to/aff2_processed \
	--cache_dir /path/to/cached_data \
	--lmdb_label_dir /path/to/LMDB \
	--exp_dir ./experiments/super/hrformer_au
```

`train2.py` and `train3.py` follow the same structure but load defaults from different `opts*.py` presets.

### 2) Two-stage supervised + Mean Teacher

Use the provided wrapper:

```bash
python train_all.py \
	--stage1_exp_dir ./exp_dir/hrformer_au \
	--stage2_exp_dir ./exp_dir/hrformer_au_mt \
	--train_args "--root /path/to/aff2_processed --cache_dir /path/to/cached_data --lmdb_label_dir /path/to/LMDB" \
	--mt_args "--root /path/to/aff2_processed --cache_dir /path/to/cached_data --lmdb_label_dir /path/to/LMDB"
```

Optional multi-GPU mode via wrapper:

```bash
python train_all.py --ddp --devices 0,1,2,3 --hardcode_batch
```

### 3) Direct Mean Teacher training

```bash
python train_mt.py \
	--task AU \
	--model_name hrformer \
	--root /path/to/aff2_processed \
	--cache_dir /path/to/cached_data \
	--lmdb_label_dir /path/to/LMDB \
	--exp_dir ./experiments/mt/hrformer_au_mt
```

For distributed training, use `train_mtmm.py` with `torchrun`.

## Evaluation and Visualization

- `test_aff2.py` / `test_aff2mm.py`: AU inference and F1 calculation
- `show_loss.py`: parse and visualize training logs
- `show_result.py`: prediction/result analysis
- `show_CAM.py` / `show_CAMmm.py`: CAM and qualitative visualization

## Important Caveats

1. Several scripts still contain hard-coded absolute paths from the original research environment (Linux and Windows drives). Always pass your own paths through CLI arguments or edit defaults in `opts*.py`.
2. Some model files include optional hard-coded pretrained checkpoint paths. If those files are missing, run without pretraining flags or replace paths.
3. Inference scripts (`test_aff2*.py`) define `model_path` near the top; set it to your checkpoint.
4. `UESTC-Thesis-Latex-Template-main/` is ignored at the repository root and is intended as local reference material, not part of the GitHub upload.

## Thesis Project Folder

`UESTC-Thesis-Latex-Template-main/` is an optional local reference folder, including:

- thesis source (`main.tex`),
- bibliography and class files,
- beamer slides template under `slides/`.

Use that folder independently for document compilation and thesis writing. It is excluded from the Git repository by root `.gitignore`.

## Acknowledgements

This project builds on ideas/codebases from:

- [Former-DFER](https://github.com/zengqunzhao/Former-DFER)
- [Two-Stream Aural-Visual Affect Analysis in the Wild](https://github.com/kuhnkeF/ABAW2020TNT)

## License

Please refer to the corresponding upstream projects and, if used locally, the thesis template folder for license details of reused components.


