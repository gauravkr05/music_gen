# Text-to-Music with fine tune LLM driven by Diffusion Model

## Overview

This project generates music from a plain-language prompt by splitting the task into two stages. First, a fine-tuned language model turns a free-text request like *"lofi for studying while it's raining at night"* into a clean, structured JSON description (genre, mood, tempo, instruments, and more). That JSON is then turned into a condition vector that drives a diffusion model, which generates the audio as a mel spectrogram.

The interesting part of the design is the middle layer: instead of feeding raw text straight into an audio model, the prompt is first compiled into a structured, validated representation. This makes the conditioning explicit and controllable, and it's the piece that separates this project from a standard text-to-audio pipeline.

## Problem Statement

Most text-to-music systems map a sentence directly to audio, which makes the conditioning a black box — you can't see or edit what the model "understood." A vague prompt produces unpredictable results, and there's no structured handle to adjust one attribute (say, the tempo or whether there are vocals) without rewriting the whole sentence.

The goal here is to put a structured layer in between: parse the prompt into a fixed schema, validate it, and condition the audio model on that schema. This gives a readable, editable description of the music before a single sample of audio is generated.

## Methodology

### The structured representation

Every prompt is converted into a JSON object with a fixed set of keys:

```json
{
  "genre": "lo-fi hip hop",
  "mood": ["calm", "nostalgic", "focused"],
  "tempo_bpm": 75,
  "intensity": 0.2,
  "vocals": "none",
  "instruments": ["electric piano", "soft drums", "rain ambience"],
  "context": {"time_of_day": "night", "environment": "a cozy bedroom",
              "weather": "rain", "activity": "studying"},
  "tags": ["study", "rain", "cozy"]
}
```

The schema is deliberately small (8 keys). Continuous fields use 0–1 floats, categorical fields use a controlled vocabulary, and the model is trained to *infer* unstated details — for example, "rain" in the prompt adds a rain-ambience layer, and "studying" pushes toward instrumental and low intensity.

### Audio as a 2D image

Audio is a 1D waveform, which is hard to model directly. Following common practice in audio generation, each clip is converted into a **mel spectrogram** — a 2D time-frequency image — so it can be handled with the same convolutional tools used for pictures.

```python
import librosa
y, sr = librosa.load("clip.wav", sr=16000)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)   # log scale
```

Every clip becomes a fixed `128 x 960` matrix, normalized to the `[-1, 1]` range so it matches the noise scale used by the diffusion model. To turn a generated spectrogram back into sound, a Griffin-Lim vocoder is used (`librosa.feature.inverse.mel_to_audio`).

## Model Architecture

### 1. Prompt to JSON (fine-tuned LLM)

File: `finetune_music_llama.ipynb`, `pipeline.py`

A **Llama 3.2 3B Instruct** model is fine-tuned with **LoRA** (rank 16, via Unsloth) on ~2,500 prompt→JSON pairs. Training masks the loss to the JSON output only, so the model learns to produce structure rather than echo the prompt. The dataset is generated synthetically with coherence rules (e.g. a gym prompt never maps to a sleep genre) and conservative defaults for vague prompts.

### 2. Validation and repair

File: `validator.py`, `render.py`

A small language model will occasionally produce malformed JSON or invent keys. A validation layer checks every output against the schema, and a repair step strips unknown keys, fixes out-of-range numbers, and normalizes bad values before anything downstream uses it. A separate render step compiles the JSON into a short text prompt for the audio model — kept out of the LLM training so the model never learns to generate prose.

### 3. Dataset generation (MusicGen)

File: `build_dataset.py`, `make_labels.py`

To get paired `(condition, audio)` data for the diffusion model, the JSON labels are fed through Meta's **MusicGen** (`musicgen-small`) to produce 15-second clips, which are then converted to spectrograms. The label is the same JSON, so every spectrogram comes with a known condition for free. This is effectively a distillation setup — the diffusion model learns to reproduce MusicGen's output conditioned on the structured JSON.

### 4. Conditional diffusion U-Net (from scratch)

File: `Music_Gen.ipynb`, `diffusion.py`

A **conditional DDPM** is implemented from scratch:

- **Forward process** gradually adds Gaussian noise to a spectrogram over 1,000 steps until it's pure noise (`add_noise`).
- **U-Net** (encoder-decoder with skip connections) predicts the noise that was added, given the noisy spectrogram, the timestep, and the **604-dimensional condition vector** built from the JSON. The condition and timestep are injected into the bottleneck.
- **Reverse process** starts from random noise and denoises step by step, steered by the condition, into a spectrogram (`sample`).

Trained with **MSE loss** between predicted and true noise.

## Training

| | LLM stage | Diffusion stage |
|---|---|---|
| **Data** | ~2,500 prompt → JSON pairs (synthetic) | ~1,000 spectrograms across 8 genres |
| **Input** | natural-language prompt | noisy spectrogram + timestep + condition |
| **Target** | structured JSON | the noise added at that step |
| **Loss** | next-token (response-only) | MSE(predicted noise, true noise) |
| **Method** | LoRA fine-tune (Unsloth) | DDPM from scratch (PyTorch) |
| **Hardware** | Kaggle / Colab T4 | Kaggle T4 |

The diffusion dataset is restricted to 8 genres (lo-fi, ambient, trap, synthwave, neoclassical piano, acoustic folk, deep house, cinematic) so the limited data is concentrated, which gives the model more examples per category to learn from.

## Applications

- A controllable text-to-music tool where the structured layer can be edited before generation.
- Mood- or scene-based background music for study, focus, games, or video.
- A teaching example of how to chain a structured-output LLM with a generative audio model.
- A reusable prompt→JSON parser that could drive any downstream audio backend, not just this one.

## Output

- `spec_dataset/mels/` — the spectrogram matrices (`.npy`) used for training.
- `spec_dataset/wavs/` — a sample of the original MusicGen clips, kept for A/B comparison.
- `checkpoints/best_model.pth` — the trained diffusion model.
- `generated.wav` + `gen.png` — a generated clip and its spectrogram from the final cell.

## Results / Observations

- The **LLM stage works well.** It parses clean prompts reliably and generalizes to unusual ones (e.g. "Shiv Tandav at late night" → Indian classical; "late night staring at the stars" → ambient). Genre and mood are usually right; it occasionally misreads scene words or under-rates intensity.
- The **diffusion stage is a proof of concept.** A from-scratch model trained on ~1,000 clips produces audio with recognizable genre character but rough, blurry quality — it is not clean music. This is expected at this data and model scale; the architecture is complete and correct, the fidelity is simply limited by data and compute.
- The validation/repair layer matters in practice — small models reliably produce the occasional malformed or hallucinated field, and catching those before generation keeps the pipeline stable.

## How to Run

```
# 1. Fine-tune the LLM (or load the trained LoRA adapter)
finetune_music_llama.ipynb

# 2. Build the spectrogram dataset (JSON -> MusicGen -> mel)
make_labels.py + build_dataset.py

# 3. Train the diffusion model
Music_Gen.ipynb     # reads spec_dataset, trains the conditional U-Net

# 4. Generate
# prompt -> LLM -> JSON -> condition -> diffusion -> spectrogram -> Griffin-Lim -> wav
```

The full inference path is: **prompt → fine-tuned LLM → JSON → validate/repair → condition vector → diffusion U-Net → spectrogram → audio.**

## Future Work

1. **Scale the diffusion data.** The biggest lever for quality is more spectrograms (10k+) and more examples per genre. The pipeline already supports appending and resuming.
2. **Better vocoder.** Replace Griffin-Lim with a trained neural vocoder (e.g. HiFi-GAN) for cleaner reconstruction from spectrograms.
3. **Faster sampling.** Swap the 1,000-step DDPM sampler for DDIM to cut generation time.
4. **Richer schema.** Re-introduce dropped fields (brightness, warmth, key) once there's enough data to learn them consistently.
5. **Cultural and scene coverage.** Add training prompts for festivals, cultural genres, and abstract scenes, where the LLM currently has the weakest mappings.
