# ESKMeans

Unsupervised speech segmentation using energy-based k-means clustering. Given speech audio, the system detects acoustic landmarks and clusters them into acoustic units without supervision, using a graph-based shortest-path algorithm for segmentation.

Supports MFCC and neural features (HuBERT, multilingual HuBERT, WavLM).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For neural feature extraction only:
pip install librosa transformers torch
```

## Data preparation

Use `prepare_dataset.py` to extract features and landmarks from WAV files:

```bash
export ESKMEANS_DATA=/path/to/data

# MFCC features
python prepare_dataset.py \
    --wav_dir data/wavs/s01 \
    --language buckeye \
    --speaker s01 \
    --feature_type mfcc

# HuBERT features
python prepare_dataset.py \
    --wav_dir data/wavs/s01 \
    --language buckeye \
    --speaker s01 \
    --feature_type hubert_base_ls960 \
    --layer 10 \
    --device cuda
```

The script writes the directory layout expected by the main pipeline:

```
$ESKMEANS_DATA/
  zerospeech_seg/mfcc_herman/<language>/<speaker>/
    landmarks.pkl
    raw_mfcc.npz
  hubert_data/seg/zsc/<feature_type>/<layer>/<language>/prevad/
    <speaker>_features_frame.npz
```

## Running

```bash
python run.py \
    --language buckeye \
    --speaker s01 \
    --feature_type hubert_base_ls960 \
    --pooling_type average \
    --kmeans_type em
```

### Arguments

| Argument | Options | Description |
|---|---|---|
| `--language` | `buckeye`, `mandarin` | Dataset / language |
| `--speaker` | e.g. `s0101a` | Speaker ID |
| `--feature_type` | `mfcc`, `hubert_base_ls960`, `mhubert`, `wavlm_large` | Feature representation |
| `--pooling_type` | `herman`, `average` | How features are aggregated between landmarks |
| `--kmeans_type` | `herman`, `em` | K-means variant |

### Environment variables

| Variable | Description |
|---|---|
| `ESKMEANS_DATA` | Base directory for data. Falls back to hardcoded scratch paths on known hosts. |

### Batch processing

```bash
cat data/file_list/buckeye_spk | parallel \
    python run.py --speaker {} --language buckeye \
    --feature_type hubert_base_ls960 \
    --pooling_type average --kmeans_type em
```

## Output

Results are written to `results/<feature_type>_<pooling_type>_<language>/<speaker>.tdev`. Each line lists a cluster ID with the utterance ID and start/end times (in centiseconds) of segments assigned to that cluster.

## Key parameters

Configured at the top of `run.py`:

| Parameter | Default | Description |
|---|---|---|
| `min_edges` | 0 | Minimum landmark gaps per segment |
| `max_edges` | 6 | Maximum landmark gaps per segment |
| `min_duration` | 20 | Minimum segment duration (centiseconds) |
| `nepochs` | 10 | Maximum EM iterations |

## Project structure

```
eskmeans.py        # Core segmentation (graph, shortest-path, k-means loop)
eskmeans_init.py   # Cluster initialisation
centroids.py       # Centroid update logic (standard and EM variants)
data_utils.py      # Dataset loading and result writing
pooling.py         # Feature pooling between landmarks
run.py             # Entry point
prepare_dataset.py # Feature extraction from raw WAV files
unit_test.py       # Validation routines
kaldiark.py        # Kaldi ARK file format reader
```
