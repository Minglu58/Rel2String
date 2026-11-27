# Rel2String

Rel2String generates upper-body, finger, and bow motion for cello and violin from synchronized audio and known string geometry.

https://github.com/user-attachments/assets/c57a512f-8826-42d2-a941-7b600ce0c8f8

## 1. Environment

```
git clone https://github.com/Minglu58/Rel2String.git
cd Rel2string
conda create -n Rel2string python==3.8
conda activate Rel2string
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
% pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```

## 2. Prepare Data and Audio Features

### 2.1 Download the original dataset

You should download the original String Performance Dataset (SPD) [here](https://metaverse-ai-lab-thu.github.io/String-Performance-Dataset-SPD/) under `datasets/` folder.

The data structure is like:

```
datasets/SPD/
├── cello/
│   ├── kp3d/cello01.npy
│   └── audio/cello01.wav
└── violin/
    ├── kp3d/violin01.npy
    └── audio/violin01.wav
```

### 2.2 Preprocess the data

You can run the following command to obtain the processed data.

```
python data_process/prepare_spd_canonical_dataset.py \
  --spd-root datasets/SPD \
  --output-root datasets/SPD_canonical \
  --split-dir splits/spd \
  --fps 30 --origin nut --scale median_nut_to_bridge \
  --audio-mode copy
```

Append `--dry-run` to check that the fixed recording lists have matching audio and 3D files before writing the dataset.

Output under `datasets/SPD_canonical/` | Contents |
| --- | --- |
| `samples/<instrument>/<take>/audio.wav` | Copy of the synchronized WAV input |
| `samples/<instrument>/<take>/string_frame.npz` | Body, bow, string and contact points in the canonical frame, plus the transform back to world coordinates |
| `samples/<instrument>/<take>/playable_relations.npz` | 23 relation channels, their validity masks, and the separately stored `bow_root_bridge_distance` target used for the 24th channel in the final model |
| `samples/<instrument>/<take>/metadata.json` | Recording ID, split, frame count, and input and output paths |
| `manifest.json`, `manifest.csv`, and `splits/{train,val,test}.{json,txt}` | Dataset index and fixed recording lists

### 2.3 Cache audio features

Place the [MERT-v1-330M snapshot](https://huggingface.co/m-a-p/MERT-v1-330M/tree/main), including its weights, model code, and processor configuration, in `checkpoints/MERT-v1-330M/`. Run the cache script after building `SPD_canonical` and before starting Stage 1, Stage 2, or evaluation.

```
python scripts/precompute_audio_features.py --config configs/spd/precompute_mert.yaml
```

`precompute_mert.yaml` reads the training, validation and test lists and writes one `.npz` per recording to `datasets/SPD_canonical/audio_feature_cache/mert/`. The cache contains MERT embeddings, mel, onset, pitch and four bow dynamics channels aligned to the 30 fps motion frame grid. A cache built by this command already includes the dynamics channels.

## 3. Download pretrained model weights

Place the pretrained model files under `checkpoints/` at the paths used by the active YAML. Stage-2 training initializes the pose UNet from scratch, and evaluation loads the Stage-2 checkpoint listed below.

Asset | Path or contents |
| --- | --- |
| MERT audio encoder for feature precomputation | `checkpoints/MERT-v1-330M/` |
| Frozen Stage 1 | `checkpoints/mapping.pt` |
| Final Stage-2 directory | `checkpoints/checkpoint-100000/` |

## 4. Inference

You can use the pretrained weights above to reproduce the results.

```
python scripts/evaluate.py --config configs/spd/test.yaml
```

The default evaluates cello and writes to `results/test_cello/`. Evaluate violin separately to obtain the instrument-specific metrics.

```
python scripts/evaluate.py --config configs/spd/test.yaml \
  test_data.instruments=violin output_dir=./results/test_violin
```

## 5. Retrain Stage 2

`configs/spd/train_stage2.yaml` contains the full recipe. Its top-level fields set the frozen checkpoint, run paths, and training budget. The inline `stage2_config` section holds the data, model, losses, optimizer, and validation settings.

```bash
python scripts/train_stage2.py --config configs/spd/train_stage2.yaml
```

The final checkpoint is written to `output_dir: results/stage2/`.

## Acknowledgements

