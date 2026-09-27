"""Select LibriSpeech test-clean utterances for experiment 011 and write 16 kHz PCM16 WAVs.

Run with the parakeet-stt venv (needs soundfile for FLAC):
  ..\\parakeet-stt\\.venv\\Scripts\\python experiments\\011_prepare_data.py

Speakers are split deterministically: 30 speakers for evaluation (7 utterances
each, spread over the speaker's list so lengths vary), the other 10 for
calibration of static quantization (6 utterances each). The two sets share no
speaker. Output: cache/parakeet/librispeech_wav/*.wav, cache/parakeet/eval.json,
cache/parakeet/calib.json.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cache" / "librispeech" / "LibriSpeech" / "test-clean"
OUT = ROOT / "cache" / "parakeet"
WAV = OUT / "librispeech_wav"


def utterances():
    for trans in sorted(SRC.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            yield uid.split("-")[0], uid, trans.parent / f"{uid}.flac", text


def pick(items, n):
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def main():
    by_speaker = {}
    for spk, uid, path, text in utterances():
        by_speaker.setdefault(spk, []).append((uid, path, text))
    speakers = sorted(by_speaker, key=int)
    calib_spk = speakers[::4]  # every 4th speaker -> 10 of 40
    WAV.mkdir(parents=True, exist_ok=True)
    sets = {"eval": [], "calib": []}
    for spk in speakers:
        name, n = ("calib", 6) if spk in calib_spk else ("eval", 7)
        for uid, path, text in pick(by_speaker[spk], n):
            audio, rate = sf.read(path, dtype="int16")
            assert rate == 16000 and audio.ndim == 1
            dst = WAV / f"{uid}.wav"
            sf.write(dst, audio, rate, subtype="PCM_16")
            sets[name].append({"id": uid, "wav": str(dst.relative_to(ROOT)), "seconds": len(audio) / rate,
                               "text": text})
    for name, items in sets.items():
        (OUT / f"{name}.json").write_text(json.dumps(items, indent=1))
        secs = np.array([x["seconds"] for x in items])
        print(f"{name}: {len(items)} utterances, {len({x['id'].split('-')[0] for x in items})} speakers, "
              f"{secs.sum() / 60:.1f} min, length min/median/max {secs.min():.1f}/{np.median(secs):.1f}/{secs.max():.1f} s")


if __name__ == "__main__":
    main()
