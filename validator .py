"""validator.py — validate V2 (lean, 8-key) music JSON from the model.
"""
import json, re

VOCALS = {"none", "male", "female", "choir", "vocal-chops", "spoken"}
REQUIRED = {"genre", "mood", "tempo_bpm", "intensity", "vocals", "instruments", "context", "tags"}
CONTEXT_KEYS = {"time_of_day", "environment", "weather", "activity"}


def extract_json(text):
    if isinstance(text, dict):
        return text
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s:
        raise ValueError("no JSON object found")
    return json.loads(text[s:e + 1])


def _str(x): return isinstance(x, str) and x.strip() != ""
def _strlist(x): return isinstance(x, list) and len(x) > 0 and all(isinstance(i, str) for i in x)


def validate_record(obj):
    errs = []
    if not isinstance(obj, dict):
        return False, ["top-level is not an object"]
    miss = REQUIRED - set(obj)
    extra = set(obj) - REQUIRED
    if miss: errs.append(f"missing keys: {sorted(miss)}")
    if extra: errs.append(f"unexpected keys: {sorted(extra)}")

    if "genre" in obj and not _str(obj["genre"]):
        errs.append("'genre' must be a non-empty string")
    for k in ("mood", "instruments"):
        if k in obj and not _strlist(obj[k]):
            errs.append(f"'{k}' must be a non-empty string array")
    if "tags" in obj and not (isinstance(obj["tags"], list) and all(isinstance(i, str) for i in obj["tags"])):
        errs.append("'tags' must be a string array")
    if "tempo_bpm" in obj:
        v = obj["tempo_bpm"]
        if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= 300):
            errs.append(f"'tempo_bpm' must be an int in [0,300] (got {v!r})")
    if "intensity" in obj:
        v = obj["intensity"]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0):
            errs.append(f"'intensity' must be a number in [0.0,1.0] (got {v!r})")
    if "vocals" in obj and obj["vocals"] not in VOCALS:
        errs.append(f"'vocals' must be one of {sorted(VOCALS)} (got {obj['vocals']!r})")
    if "context" in obj:
        c = obj["context"]
        if not isinstance(c, dict):
            errs.append("'context' must be an object")
        else:
            cm = CONTEXT_KEYS - set(c)
            if cm: errs.append(f"'context' missing: {sorted(cm)}")
            cx = set(c) - CONTEXT_KEYS
            if cx: errs.append(f"'context' has unexpected keys: {sorted(cx)}")
            for sk in ("time_of_day", "environment"):
                if sk in c and not _str(c[sk]):
                    errs.append(f"'context.{sk}' must be a non-empty string")
            for sk in ("weather", "activity"):
                if sk in c and c[sk] is not None and not _str(c[sk]):
                    errs.append(f"'context.{sk}' must be a string or null")
    return (len(errs) == 0), errs



KNOWN_WEATHER = {None, "rain", "storm", "snow", "sunny", "cloudy", "foggy", "windy", "clear"}


def repair_record(obj):
    """Coerce a parsed object toward the schema WITHOUT inventing missing data:
    - drop unknown top-level keys and unknown context keys (e.g. hallucinated
      'temperature'/'humidity')
    - null out a weather value that isn't a real weather word (e.g. 'star')
    - clamp intensity to [0,1] and tempo_bpm to [0,300]
    Returns a new dict. Run this BEFORE validate_record."""
    if not isinstance(obj, dict):
        return obj
    out = {k: obj[k] for k in REQUIRED if k in obj}
    c = out.get("context")
    if isinstance(c, dict):
        c = {k: c[k] for k in CONTEXT_KEYS if k in c}
        w = c.get("weather")
        if isinstance(w, str):
            wl = w.strip().lower()
            SEASON_MAP = {"summer": "sunny", "winter": "snow", "spring": "sunny", "autumn": "cloudy", "fall": "cloudy"}
            if wl in SEASON_MAP:
                c["weather"] = SEASON_MAP[wl]
            elif wl not in {x for x in KNOWN_WEATHER if x}:
                c["weather"] = None
        out["context"] = c
    if isinstance(out.get("intensity"), (int, float)) and not isinstance(out.get("intensity"), bool):
        out["intensity"] = round(max(0.0, min(1.0, out["intensity"])), 2)
    if isinstance(out.get("tempo_bpm"), int) and not isinstance(out.get("tempo_bpm"), bool):
        out["tempo_bpm"] = max(0, min(300, out["tempo_bpm"]))
    return out


def validate_text(text):
    try:
        obj = extract_json(text)
    except Exception as e:
        return False, [f"JSON parse failed: {e}"], None
    ok, errs = validate_record(obj)
    return ok, errs, obj


def report_file(path):
    total = good = 0; fails = []
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            payload = json.loads(line)["messages"][-1]["content"]
            total += 1; ok, errs, _ = validate_text(payload)
            good += ok
            if not ok and len(fails) < 5: fails.append((errs, payload[:120]))
    else:
        import csv
        r = csv.reader(open(path, encoding="utf-8")); next(r, None)
        for row in r:
            if not row: continue
            total += 1; ok, errs, _ = validate_text(row[1]); good += ok
            if not ok and len(fails) < 5: fails.append((errs, row[1][:120]))
    return total, good, fails


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "music_prompts.jsonl"
    t, g, fails = report_file(p)
    print(f"{g}/{t} valid ({100*g/t:.2f}%)")
    for e, s in fails: print("  FAIL:", e, "|", s)
