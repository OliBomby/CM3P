# CM3P: Contrastive Metadata-Map Masked Pre-training

CM3P (Contrastive Metadata-Map Masked Pre-training) is a multi-modal representation learning framework for osu! beatmaps. It learns high-quality embeddings for both beatmap structure (events, timing, positions, hitsounds, scroll speed, etc.) and beatmap metadata (difficulty, year, mapper, tags, mode, etc.), optionally conditioned on audio. These embeddings serve as a foundation for downstream tasks such as beatmap retrieval, recommendation, classification (e.g. ranked vs unranked), masked modeling, and transfer to fine-tuned generative or discriminative models.

CM3P provides:
- Unified multi-modal processor: parses raw `.osu` files + metadata + audio into token & feature tensors.
- Dual-tower ModernBERT encoders (beatmap + metadata) with optional fused audio embeddings via placeholder audio tokens.
- Contrastive embedding pretraining with structured metadata variations (robust in-batch negatives).
- Optional masked beatmap language modeling and downstream classification heads.
- High-quality embeddings for retrieval, recommendation, filtering, and fine-tuning bases.
- Flexible Hydra configuration & Hugging Face Trainer integration (freeze/unfreeze, Muon optimizer, WandB & Hub push).
- Efficient long sequence handling (Flash Attention 2 support) and mixed precision.

---
## 1. Quick Start (Inference)

To use a CM3P model in your project, you can simply load it from [Hugging Face Hub](https://huggingface.co/OliBomby/CM3P) and start extracting embeddings:
```python
import torch
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "OliBomby/CM3P"

processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True, revision="main")
model = AutoModel.from_pretrained(repo_id, device_map=device, dtype=torch.bfloat16, trust_remote_code=True, revision="main")

inputs = processor(beatmap="path/to/beatmap.osu", audio="path/to/audio.mp3")
inputs = inputs.to(device, dtype=torch.bfloat16)

with torch.no_grad():
    outputs = model(**inputs)

beatmap_embeds = outputs.beatmap_embeds  # (beatmap_length_seconds / 16, projection_dim)
```

---
## 2. Installation
### Prerequisites
- Python 3.12
- [Git](https://git-scm.com/downloads)
- [ffmpeg](http://www.ffmpeg.org/)
- [CUDA](https://developer.nvidia.com/cuda-zone) (For NVIDIA GPUs) or [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) (For AMD GPUs on linux)
- [PyTorch](https://pytorch.org/get-started/locally/): Make sure to follow the Get Started guide so you install `torch` and `torchaudio` with GPU support. Select the correct Compute Platform version that you have installed in the previous step.
- A GPU for efficient training (Flash Attention 2 support recommended). For CPU-only or unsupported GPUs, set `attn_implementation: sdpa`

### Steps
```bash
# Clone the repository
git clone https://github.com/OliBomby/CM3P.git
cd CM3P

# (Optional) Create and activate a virtual environment
python -m venv .venv

# In cmd.exe
.venv\Scripts\activate.bat
# In PowerShell
.venv\Scripts\Activate.ps1
# In Linux or MacOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention 2 if your GPU supports it (A100/H100/RTX 40xx, etc.)
# Follow official instructions; otherwise switch attn implementation in config.
```

---
## 3. Data Preparation

Create your own dataset using the [Mapperator console app](https://github.com/mappingtools/Mapperator/blob/master/README.md#create-a-high-quality-dataset). It requires an [osu! OAuth client token](https://osu.ppy.sh/home/account/edit) to verify beatmaps and get additional metadata. Place the dataset in a `datasets` directory next to the `Mapperatorinator` directory.

```sh
Mapperator.ConsoleApp.exe dataset2 -t "/Mapperatorinator/datasets/beatmap_descriptors.csv" -i "path/to/osz/files" -o "/datasets/cool_dataset"
```

When training CM3P, you can provide multiple dataset roots (list of paths) in `configs/train/default.yaml` under `dataset.train_dataset_paths` and `dataset.test_dataset_paths`.

Filtering knobs (see `dataset` section in config):
- Year range (`min_year`, `max_year`)
- Difficulty range (`min_difficulty`, `max_difficulty`)
- Gamemodes filter (`gamemodes` list)
- Splitting via indices (`train_dataset_start`, `train_dataset_end`, etc.)

---
## 4. Model Architecture
CM3P consists of three main transformer-based components built on ModernBERT:
- **Metadata Tower** (`CM3PMetadataTransformer`): Encodes metadata token sequences; pools either CLS token or mean over valid tokens.
- **Beatmap Tower** (`CM3PBeatmapTransformer`): Encodes beatmap token sequences; internally can fuse audio embeddings by replacing audio placeholder tokens with projected audio features produced by:
  - **Audio Encoder** (`CM3PAudioEncoder`): Two 1D convolutional layers (inspired by Whisper) + ModernBERT + projection MLP (`CM3PMultiModalProjector`) to reach the same embedding dimensionality as beatmap token embeddings.
- **Projection Heads**: Linear layers map pooled outputs of both towers into a shared `projection_dim` embedding space.

Optional components:
- **Masked LM Head** (`CM3PPredictionHead + decoder`): When `has_decoder_head=True` in config, produces logits over beatmap vocabulary for MLM training/inference.

### Objectives
- **Contrastive Loss** (`cm3p_loss`): Symmetric cross-entropy over similarity matrices between beatmap embeddings and metadata embeddings. If metadata variations are present, the original metadata acts as the positive; others as structured negatives.
- **Masked LM Loss** (if enabled): Standard token-level cross-entropy over masked positions.
- **Classification Loss** (downstream fine-tunes): For tasks like ranked vs unranked beatmap classification (`CM3PForBeatmapClassification`).

### Attention Implementation

`attn_implementation` can be set to `flash_attention_2` (for supported GPUs with Flash Attention 2 installed), `sdpa` (standard PyTorch attention), or `eager` (fallback implementation).
Flash Attention 2 offers significant speed and memory benefits for long sequences. CM3P + Flash Attention also supports unpadding batched input sequences for token efficiency.

---
## 5. Training (From Scratch / Fine-tuning)
### Base Command
Hydra is used for config composition:
```bash
python train.py --config-name v1              # Uses configs/train/v1.yaml -> defaults chain
python train.py --config-name v7              # Swap to another experiment variant
```
Override any field inline:

```bash
python train.py -cn v7 training.learning_rate=5e-5 dataset.labels=masked_lm model_cls=CM3PForMaskedLM
```

For all overridable configurations see `configs/train/default.yaml`.
I recommend making a copy of an existing config (e.g., `v7.yaml`) and modifying it for your experiments.

### Fine-tuning From Pretrained
Provide a checkpoint path or load a Hub model:

```bash
python train.py -cn "v7_classifier" pretrained_path="OliBomby/CM3P" dataset={train_dataset_paths:["MMRS39389","MMUS40000"],test_dataset_paths:["MMRS39389","MMUS40000"]} training={logging_steps:10,dataloader_num_workers:8} wandb_entity=mappingtools
```

### Resume Training
If `output_dir` has checkpoints and `overwrite_output_dir=false`, the script auto-resumes (unless overridden by `training.resume_from_checkpoint`).

### WandB Logging
Set:
```bash
wandb_project=CM3P wandb_entity=your_entity wandb_mode=online
```
Disable (offline) by `wandb_mode=disabled` or remove variables.

### Pushing to Hugging Face Hub
Make sure you are logged in (`huggingface-cli login`).
Enable:
```bash
training.push_to_hub=true
```
Or  use `push_to_hub.py` after training with a path to the saved checkpoint.

---
## 6. Evaluation & Metrics
`compute_metrics` (in `train.py`) aggregates metrics across evaluation steps:
- Zero-shot classification accuracy per variation class: original vs altered year/status/tags/mapper.
- Masked LM accuracy (if MLM labels present).
- Classification accuracy + top-5 for beatmap-level tasks.

During evaluation, metadata variation groups are resolved to check whether the highest-scoring metadata among variations is the original.

Metrics logged to console, saved to `eval_results.json` style files in `output_dir`, and optionally to WandB.

---
## 7. Configuration Overview
`configs/train/default.yaml` controls training, processor parameters, dataset filtering, and Hydra output directory.
`configs/model/` contains model-level defaults (can extend for different projection dims, hidden sizes, enabling decoder heads, etc.).

To inspect active config at runtime, print or log `OmegaConf.to_yaml(args)` (you can add a line in `train.py`).

---
## 8. Advanced Topics
### Audio Fusion Details
Audio features are extracted (log-mel), chunked to `max_source_positions`, passed through conv + ModernBERT encoder, then projected. The resulting dense embeddings replace placeholder audio tokens in the beatmap embedding sequence before the beatmap transformer processes them.

### Metadata Variations
Multiple metadata sequences per beatmap allow structured negatives (e.g., one with altered tags/year). Loss only treats the original (`variation class 0`) as positive; others increase robustness.

---
## 9. Troubleshooting
- OOM: Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, lower sequence/window lengths (`processor.default_kwargs.beatmap_kwargs.max_length`).
- Slow data loading: Increase `training.dataloader_num_workers` or reduce `cycle_length`.

---
## 10. Roadmap / Next Steps
- Colab notebook examples for inference & embedding extraction.
- Add evaluation suite for embedding quality (e.g., tag clustering, similarity ranking).
- Evaluate beatmap generative models using distributions of CM3P embeddings.

## Related works

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) by Syps (Nick Sypteras)
2. [osumapper](https://github.com/kotritrona/osumapper) by kotritrona, jyvden, Yoyolick (Ryan Zmuda)
3. [osu-diffusion](https://github.com/OliBomby/osu-diffusion) by OliBomby (Olivier Schipper), NiceAesth (Andrei Baciu)
4. [osuT5](https://github.com/gyataro/osuT5) by gyataro (Xiwen Teoh)
5. [Beat Learning](https://github.com/sedthh/BeatLearning) by sedthh (Richard Nagyfi)
6. [osu!dreamer](https://github.com/jaswon/osu-dreamer) by jaswon (Jason Won)
7. [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) by OliBomby (Olivier Schipper)