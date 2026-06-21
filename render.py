
# render.py — turn the predicted JSON into a text prompt for the audio model.
# Two builders:
  

def _tempo_words(bpm):
    if not bpm: return "beatless, very slow"
    if bpm < 70:  return "slow"
    if bpm < 95:  return "laid-back, mid-tempo"
    if bpm < 120: return "upbeat"
    if bpm < 135: return "driving"
    return "fast, high-energy"


def _intensity_words(x):
    if x is None: return ""
    if x < 0.3:  return "gentle and sparse"
    if x < 0.6:  return "moderate energy"
    if x < 0.8:  return "energetic"
    return "intense and powerful"


_WX = {"rain": "rainy", "storm": "stormy", "snow": "snowy", "sunny": "sunny",
       "cloudy": "cloudy", "foggy": "foggy", "windy": "windy"}


def build_render_prompt(obj):
    """Rich descriptive sentence (general purpose)."""
    genre = obj.get("genre", "music")
    mood = obj.get("mood", []) or []
    bpm = obj.get("tempo_bpm", 0)
    vocals = obj.get("vocals", "none")
    instruments = obj.get("instruments", []) or []
    ctx = obj.get("context", {}) or {}
    intensity = obj.get("intensity")

    tempo = _tempo_words(bpm)
    bpm_txt = "" if not bpm else f" around {bpm} BPM"
    voc = "fully instrumental" if vocals == "none" else f"with {vocals} vocals"
    instr = ", ".join(instruments[:5])
    mood_txt = ", ".join(mood[:3]) if mood else "atmospheric"

    parts = [f"A {tempo} {genre} track{bpm_txt}"]
    if instr:
        parts.append(f"featuring {instr}")
    parts.append(voc)
    feel = f"{mood_txt} in feel"
    iw = _intensity_words(intensity)
    if iw:
        feel += f", {iw}"
    parts.append(feel)

    act, env = ctx.get("activity"), ctx.get("environment")
    wx, tod = ctx.get("weather"), ctx.get("time_of_day")
    bits = []
    if act:
        bits.append(f"suited for {act}")
    loc = env if env and env != "any" else None
    when = None
    if wx:
        when = f"a {_WX.get(wx, wx)} {tod}" if tod and tod != "any" else f"a {_WX.get(wx, wx)} day"
    elif tod and tod != "any":
        when = tod
    if loc and when:
        bits.append(f"set in {loc} on {when}" if wx else f"set in {loc} at {when}")
    elif loc:
        bits.append(f"set in {loc}")
    elif when:
        bits.append(f"on {when}" if wx else f"at {when}")
    if bits:
        parts.append(", ".join(bits))
    return ". ".join(parts) + "."


def to_audio_prompt(obj):
    """Concise, comma-separated prompt tuned for MusicGen.
    MusicGen responds best to musical descriptors (genre, tempo, mood, instruments),
    so we drop scene/activity context here and keep it tight."""
    genre = obj.get("genre", "music")
    parts = [genre]
    bpm = obj.get("tempo_bpm", 0)
    if bpm:
        parts.append(f"{bpm} BPM")
    mood = obj.get("mood", []) or []
    if mood:
        parts.append(", ".join(mood[:3]))
    vocals = obj.get("vocals", "none")
    instr = [i for i in (obj.get("instruments", []) or []) if "vocal" not in i.lower()]
    if instr:
        parts.append("with " + ", ".join(instr[:5]))
    parts.append("instrumental" if vocals == "none" else f"{vocals} vocals")
    return ", ".join(parts)


if __name__ == "__main__":
    demo = {"genre": "lo-fi hip hop", "mood": ["calm", "nostalgic", "focused"],
            "tempo_bpm": 75, "intensity": 0.2, "vocals": "none",
            "instruments": ["electric piano", "soft drums", "sub bass", "rain ambience"],
            "context": {"time_of_day": "night", "environment": "a cozy bedroom",
                        "weather": "rain", "activity": "studying"}, "tags": []}
    print("DESCRIPTIVE:", build_render_prompt(demo))
    print("MUSICGEN   :", to_audio_prompt(demo))
