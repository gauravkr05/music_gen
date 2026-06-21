from validator import validate_text, repair_record, validate_record
from render import to_audio_prompt, build_render_prompt

# system message MUST match the one used in training
SYSTEM_MSG = ("You convert a natural-language music request into a single valid JSON object "
              "with exactly these keys: genre, mood, tempo_bpm, intensity, vocals, instruments, "
              "context, tags. Infer only what the request reasonably implies; do not invent a "
              "genre or vocals the request does not support. For vague requests, choose a common, "
              "safe genre. Respond with JSON only.")


# ---------- stage 1: load the fine-tuned LLM ----------
def load_llm(adapter_dir="music_lora", max_seq_length=2048):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=max_seq_length,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tok


def _llm_once(user_prompt, model, tok, temperature):
    msgs = [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    out = model.generate(input_ids=ids, max_new_tokens=400,
                         do_sample=(temperature > 0),
                         temperature=max(temperature, 1e-4),
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ---------- stage 2: prompt -> validated JSON (with retry) ----------
def prompt_to_json(user_prompt, model, tok, max_retries=2):
    """Returns (obj, raw_text, errors). obj is None if all attempts fail.
    Each attempt is repaired (unknown keys stripped, bad weather nulled, values
    clamped) BEFORE validating, so benign hallucinations are auto-cleaned."""
    import json as _json
    from validator import extract_json
    temps = [0.3, 0.1, 0.0]
    last = None
    for i in range(max_retries + 1):
        raw = _llm_once(user_prompt, model, tok, temps[min(i, len(temps) - 1)])
        try:
            obj = repair_record(extract_json(raw))
        except Exception as e:
            last = (raw, [f"parse failed: {e}"]); continue
        ok, errs = validate_record(obj)
        if ok:
            return obj, raw, None
        last = (raw, errs)
    return None, last[0], last[1]


# ---------- stage 3: load MusicGen ----------
def load_audio_model(name="facebook/musicgen-small", device="cuda"):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    processor = AutoProcessor.from_pretrained(name)
    audio = MusicgenForConditionalGeneration.from_pretrained(name).to(device)
    return processor, audio


# ---------- stage 4: JSON -> .wav ----------
def json_to_audio(obj, processor, audio_model, out_path="out.wav",
                  duration_s=10, guidance_scale=3.0, device="cuda"):
    import scipy.io.wavfile
    text = to_audio_prompt(obj)                       # MusicGen-tuned prompt
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
    max_new = int(duration_s * 50)                    # MusicGen ~= 50 tokens / second (max ~30s)
    wav = audio_model.generate(**inputs, do_sample=True,
                               guidance_scale=guidance_scale, max_new_tokens=max_new)
    sr = audio_model.config.audio_encoder.sampling_rate
    data = wav[0, 0].cpu().numpy()
    scipy.io.wavfile.write(out_path, rate=sr, data=data)
    return out_path, text, sr


def generate(user_prompt, llm, tok, processor, audio_model,
             out_path="out.wav", duration_s=10):
    obj, raw, errs = prompt_to_json(user_prompt, llm, tok)
    if obj is None:
        raise ValueError(f"LLM did not return valid JSON after retries: {errs}\nraw: {raw[:200]}")
    path, atext, sr = json_to_audio(obj, processor, audio_model, out_path, duration_s)
    return {"json": obj, "render": build_render_prompt(obj),
            "audio_prompt": atext, "wav": path, "sr": sr}
