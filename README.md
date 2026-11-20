# TSformer-VO: Oxford Spires experiments
[![DOI](https://zenodo.org/badge/1289987882.svg)](https://doi.org/10.5281/zenodo.24404818)

This repository contains the TSformer-VO implementation and the Oxford Spires training and evaluation workflow used for the monocular visual-odometry experiments. The model implementation is adapted from [TimeSformer](https://github.com/facebookresearch/TimeSformer) and the original [TSformer-VO project](https://github.com/aofrancani/TSformer-VO). It is designed to work as a standalone repository or as a submodule.

Run commands from this repository's root directory (the `TSformer-VO/` directory when included as a submodule). Configuration, data, and command output paths are relative to that working directory; the Slurm launcher changes to its own repository directory before running. Slurm resolves the launcher's `log_oxford.out` path relative to the directory from which the job is submitted.

The Oxford experiment documented here fine-tunes the two-frame TSformer-VO model on ten Oxford Spires sequences. Its two test sequences, Christ Church 02 and 05, are from the same site and are held out from the Oxford fine-tuning pool. The tracked timestamp CSVs define the evaluated doorway subsequences.

## Repository contents

- `train.py` and `configs/oxford_experiment.json`: fine-tuning workflow and the final Oxford experiment configuration.
- `evaluate_tsformer_oxford.py`: inference and APE/RPE evaluation on Oxford sequences or generated subsequences.
- `copy_subset.py`: creates the doorway subsequences from the tracked timestamp CSV files.
- `plot_doorway_trajectories.py`: creates combined doorway trajectory figures from evaluation outputs.
- `datasets/` and `timesformer/`: dataset loaders and model implementation.
- `checkpoints/`, `comparison/`, and `oxford/`: local experiment outputs and data; these are intentionally excluded from Git.

## Environment

The original model checkpoints and pinned dependencies target Python 3.8 and PyTorch 1.10.1. A CUDA-capable GPU is recommended for training; evaluation can run on CPU. The batch script `train_oxford.sh` uses the repository-local `.venv`, so create and activate that environment from the repository root as follows:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`scipy` is used for pose conversion and `evo` provides the `evo_ape`, `evo_rpe`, and `evo_traj` commands used by evaluation; both are included in `requirements.txt`. Install a PyTorch build compatible with the machine's CUDA setup if the default `requirements.txt` install is not suitable. After activating `.venv`, run `bash train_oxford.sh` locally or submit it with `sbatch train_oxford.sh` on Slurm. The launcher resolves the repository directory from its own location, so it can also be submitted while the current directory is elsewhere.

## Download and prepare the data

The Oxford Spires data and model weights are not included in this repository. Start at the [official Oxford Spires dataset page](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/), follow its Hugging Face or Google Drive download link, and obtain the camera images and provided ground-truth trajectory for the twelve sequences below. The official page links the [dataset structure documentation](https://github.com/ori-drs/oxford_spires_dataset/wiki/Dataset-Structure), which describes the archive layout.

Download only the required sequences' `raw/images.zip` and `processed/trajectory/gt-tum.txt` files; the full dataset includes large LiDAR and mapping data that this experiment does not use. Extract each `images.zip` in its sequence's `raw/` directory. The downloaded sequence tree should have this structure:

```text
oxford_spires_dataset/sequences/
    2024-03-14-blenheim-palace-01/
        raw/cam0/<timestamp>.jpg
        processed/trajectory/gt-tum.txt
    ...
```

The model uses the front camera (`cam0`) and the provided TUM trajectory. To prepare all sequences used by this experiment, run this script from the TSformer-VO repository root, pointing it at the downloaded `sequences/` directory:

```bash
python prepare_oxford_5hz.py \
  --sequences-root /path/to/oxford_spires_dataset/sequences \
  --output-root oxford/5hz
```

The script processes the ten training sequences and two test sequences listed below. For each sequence it estimates the source rate as the inverse median interval between sorted image timestamps, then chooses the integer stride nearest to a 5 Hz target (`round(source_rate / 5)`, with a minimum stride of one). It selects frames at indices `stride - 1, 2 * stride - 1, ...` and requires the resulting estimated rate to be within 15% of 5 Hz. Thus a roughly 20 Hz sequence uses stride 4, while a roughly 5 Hz sequence uses stride 1. `gt-tum.txt` is copied unchanged. The `--dry-run` option reports detected rates, strides, and frame counts without writing files. Outputs are not overwritten; use a fresh output directory if preparing the data again.

| Official Oxford Spires sequence | Local name | Use |
| --- | --- | --- |
| `2024-03-14-blenheim-palace-01` | `blenheim_palace_01` | Train |
| `2024-03-14-blenheim-palace-02` | `blenheim_palace_02` | Train |
| `2024-03-14-blenheim-palace-05` | `blenheim_palace_05` | Train |
| `2024-05-20-bodleian-library-02` | `bodleian_library_02` | Train |
| `2024-03-12-keble-college-02` | `keble_college_02` | Train |
| `2024-03-12-keble-college-03` | `keble_college_03` | Train |
| `2024-03-12-keble-college-04` | `keble_college_04` | Train |
| `2024-03-12-keble-college-05` | `keble_college_05` | Train |
| `2024-03-13-observatory-quarter-01` | `observatory_quarter_01` | Train |
| `2024-03-13-observatory-quarter-02` | `observatory_quarter_02` | Train |
| `2024-03-18-christ-church-02` | `christ_church_02` | Test |
| `2024-03-20-christ-church-05` | `christ_church_05` | Test |

This creates the paths expected by `configs/oxford_experiment.json`:

```text
oxford/5hz/<local-name>/
    sequence/<timestamp>.jpg
    gt-tum.txt
```

The ten sequences marked Train form the Oxford fine-tuning pool. Both Christ Church sequences are excluded from that pool and are held out at the site level for testing.

Download the original two-frame TSformer-VO checkpoint and its `args.pkl` from the [upstream checkpoint links](https://github.com/aofrancani/TSformer-VO#2-pre-trained-models). The upstream project documents its trained TSformer-VO checkpoints in the context of KITTI; this KITTI-trained two-frame checkpoint initializes the separate Oxford fine-tuning stage described below. Place the files at:

```text
checkpoints/model1/args.pkl
checkpoints/model1/checkpoint_model1_exp12.pth
```

The checkpoint weights are not included in this repository, and this repository records no trusted upstream checksum or checkpoint DOI. Evaluation records SHA-256 fingerprints for the selected checkpoint and arguments, and (when their files are available) the initialization checkpoint and its arguments, in `run_metadata.json`. These fingerprints identify local file contents only; they do not establish provenance or authenticate the weights against an upstream release.

## Reproduce training

From the repository root, start a fresh run with:

```bash
python train.py --config configs/oxford_experiment.json
```

The config fine-tunes from the original checkpoint with Adam (`lr=1e-5`), batch size 8, and up to 100 epochs. It uses two frames per input window (`window_size=2`, `overlap=1`), 192×640 images, and a rotating sequence-level validation split over the ten training sequences. Each epoch holds out one whole training sequence for validation and trains on the other nine; the held-out sequence rotates by epoch. In this mode, the configured `val_split` fraction is not used. `checkpoint_best.pth` is selected by the lowest validation loss observed across those rotating folds. The two Christ Church test sequences are evaluated once after training and do not select the checkpoint. `train.py` sets the PyTorch seed to 2023; it does not force deterministic kernels, so exact bit-for-bit results can still depend on the PyTorch/CUDA hardware and runtime.

The fresh-run config has `"checkpoint": null`. To resume an interrupted run, set it to `"checkpoint_last.pth"` and rerun the same command. Checkpoints and TensorBoard logs are written under `checkpoints/oxford_experiment/` and are not tracked by Git. The validation-selected model is `checkpoint_best.pth`.

## Reproduce evaluation and export TUM trajectories

After preparing the full sequences, generate the two sets of doorway subsequences from the tracked timestamp CSVs:

```bash
python copy_subset.py \
  --spec "2024-03-18-christ-church-02_timestamps 1.csv" oxford/5hz/christ_church_02 church_02 \
  --spec "2024-03-20-christ-church-05_timestamps 1.csv" oxford/5hz/christ_church_05 church_05 \
  --output-root oxford/subsequences
```

To preview the selected frame counts first, add `--dry-run`. The generator does not overwrite existing subsequences; run it once into a fresh output root to avoid duplicate, suffixed folders. To evaluate the validation-selected checkpoint and write metrics plus TUM-format trajectories:

```bash
python evaluate_tsformer_oxford.py \
  --checkpoint-path checkpoints/oxford_experiment \
  --checkpoint-name checkpoint_best \
  --sequence-roots oxford/subsequences/church_02 oxford/subsequences/church_05 \
  --output-dir comparison/tsformer_eval/oxford_experiment_checkpoint_best
```

Each subsequence output folder contains `pred_tum.txt` (predicted poses), `gt_overlap.txt` (ground truth in the timestamp overlap), APE/RPE metric archives, and an XY plot. `metrics_summary.csv` records per-subsequence APE/RPE maximum and RMSE values, image/pose/GT coverage counts, nearest image-to-GT timestamp-difference summaries, and status/notes for failed or skipped evaluation steps. The loader matches each image to its nearest ground-truth timestamp without a maximum-difference cutoff; the reported timestamp differences make that matching visible without dropping samples. EVO then uses rigid trajectory alignment (`-a`, without scale correction) and the configured maximum timestamp difference (0.5 seconds by default).

Status meanings: `ok` means both metrics and the plot completed; `partial` means at least one metric completed but a metric or plot step failed; `fail` means both metric commands failed; `skip` means the input or overlap was insufficient to run evaluation. A non-`ok` row is not a complete valid measurement. Use `num_images`, `num_pred_poses`, and `num_gt_overlap` to inspect coverage, and review `notes` and timestamp-difference columns before comparing results.

The output also includes `tum_outputs_manifest.csv`, which maps each exact `bag_name` from the input timestamp CSV (the subsequence folder name) to its sequence group, status, and relative paths for the predicted and matching ground-truth TUM files. Share the listed files with downstream analysis, honoring the status. `run_metadata.json` records the resolved evaluation parameters, model parameters, runtime/package versions, device, and local-file fingerprints for the evaluated and available initialization artifacts. These generated files remain local because `comparison/` is Git-ignored.

To regenerate the combined doorway plots after evaluation:

```bash
python plot_doorway_trajectories.py
```

The plot script reads the best-checkpoint output above and the full-sequence ground truth under `oxford/5hz/`.

## Citation

This code is based on the TSformer-VO method. Please cite the original paper if you use the model:

```bibtex
@article{Francani2025,
  author={Françani, André O. and Maximo, Marcos R. O. A.},
  journal={IEEE Access},
  title={Transformer-Based Model for Monocular Visual Odometry: A Video Understanding Approach},
  year={2025},
  volume={13},
  pages={13959-13971},
  doi={10.1109/ACCESS.2025.3531667}
}
```
