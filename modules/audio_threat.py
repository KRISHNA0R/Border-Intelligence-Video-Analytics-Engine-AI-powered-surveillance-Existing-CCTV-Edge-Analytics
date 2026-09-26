"""Audio Threat Detection — heuristic analysis + clearly-labelled demo.

Honesty rules enforced here:
- No microphone is available inside a Streamlit browser session, so live
  mic capture is NOT faked. We accept an uploaded 16-bit PCM WAV and analyse
  it with numpy FFT statistics.
- Classification is a documented heuristic (impulse/crest/spectral
  features), NOT a trained sound-events model. The UI says so.
- Simulated gunshot/scream buttons create alerts marked SIMULATED.
"""
from __future__ import annotations

import io
import wave

import numpy as np


def _read_wav(data: bytes) -> float | None:
    """Extract mono float samples in [-1, 1]. None if unreadable."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            n = w.getnframes()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            fr = w.readframes(n)
        if sw != 2:
            return None
        raw = np.frombuffer(fr, dtype=np.int16)
        if raw.size == 0:
            return None
        if ch > 1:
            raw = raw.reshape(-1, ch)[:, 0]
        return raw.astype(np.float64) / 32768.0
    except Exception:
        return None


def _features(samples: np.ndarray) -> dict:
    x = samples.astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    crest = (peak / rms) if rms > 1e-9 else 0.0
    # zero-crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8))))) if x.size > 1 else 0.0
    # spectral centroid + high-band energy ratio (single 4096-pt window)
    n = min(4096, x.size)
    win = x[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(n, d=1.0 / 22050)
    total = float(spec.sum()) or 1e-9
    centroid = float((spec * freqs).sum() / total)
    high_ratio = float(spec[freqs > 2000].sum() / total)
    return {"rms": rms, "peak": peak, "crest": crest, "zcr": zcr,
            "centroid": centroid, "high_ratio": high_ratio}


def classify(features: dict) -> tuple[str, float, list[str]]:
    """Heuristic sound-class estimate. Returns (label, confidence, reasons).

    Rule set (documented, demo-grade):
      - high crest factor + high ZCR  -> sharp impulse (Possible gunshot)
      - loud + sustained + high-band  -> Scream / distress
      - otherwise                     -> Normal / ambient
    """
    r = features
    reasons = [
        f"RMS energy {r['rms']:.3f}", f"crest factor {r['crest']:.1f}",
        f"zero-crossing {r['zcr']:.2f}",
        f"high-band ratio {r['high_ratio']:.2f}",
        f"spectral centroid {r['centroid']:.0f} Hz"]
    if r["crest"] > 12 and r["zcr"] > 0.25 and r["peak"] > 0.2:
        conf = min(0.95, 0.6 + (r["crest"] - 12) * 0.02)
        return "Possible gunshot", round(conf, 2), reasons
    if r["rms"] > 0.12 and r["high_ratio"] > 0.5:
        conf = min(0.9, 0.55 + r["rms"] * 0.8)
        return "Scream / Distress Sound", round(conf, 2), reasons
    return "Normal / Ambient", 0.88, reasons


def analyze_wav(data: bytes) -> dict | None:
    samples = _read_wav(data)
    if samples is None:
        return None
    feats = _features(samples)
    label, conf, reasons = classify(feats)
    return {"label": label, "confidence": conf, "reasons": reasons,
            "features": feats}


def render_audio_threat(make_alert) -> None:
    """Audio Threat section — uploaded WAV analysis + simulated events."""
    import streamlit as st

    st.subheader("🔊 Audio Threat Detection")
    st.markdown(
        "<span class='chip chip-warn'>● HEURISTIC ENGINE</span> "
        "<span class='chip chip-info'>● NOT A TRAINED SOUND MODEL</span>",
        unsafe_allow_html=True)
    st.caption(
        "Sound classes: **Possible gunshot** · **Scream / distress** · "
        "**Normal ambient**. Classification uses documented energy/spectral "
        "heuristics, not a validated ML model. No gender claim is made for "
        "a scream. Live microphone input is not available in a browser "
        "session — upload a 16-bit PCM WAV instead."
    )

    tab_upload, tab_sim = st.tabs(["📁 Upload WAV", "🎛 Simulated Events"])

    with tab_upload:
        wav = st.file_uploader("Audio file (16-bit PCM WAV only)",
                               type=["wav"], key="audio_wav")
        if wav is not None:
            with st.spinner("Analysing audio…"):
                res = analyze_wav(wav.getvalue())
            if res is None:
                st.error("Could not decode this WAV. Use 16-bit PCM mono/stereo.")
            else:
                label = res["label"]
                sev = "critical" if label == "Possible gunshot" else (
                    "high" if label == "Scream / Distress Sound" else "low")
                icon = {"Possible gunshot": "🔴", "Scream / Distress Sound": "🟠",
                        "Normal / Ambient": "🟢"}[label]
                st.markdown(f"### {icon} {label}")
                st.markdown(f"Confidence: **{res['confidence']:.0%}**")
                for why in res["reasons"]:
                    st.caption(f"• {why}")
                if label != "Normal / Ambient":
                    make_alert(
                        "audio_threat",
                        f"AUDIO THREAT — {label} detected in uploaded audio. "
                        f"Confidence {res['confidence']:.0%}.",
                        sev, payload={"audio_label": label,
                                      "confidence": res["confidence"]},
                        confidence=res["confidence"])
                    st.success("Alert pushed to the Ledger + Alert Log.")
                else:
                    st.info("Ambient audio — no alert raised.")

    with tab_sim:
        st.caption("Simulated events are clearly marked SIMULATED — they are "
                   "not real detections.")
        c1, c2 = st.columns(2)
        if c1.button("💥 Simulate: Possible gunshot", use_container_width=True,
                     key="audio_sim_gun"):
            make_alert(
                "audio_threat",
                "AUDIO THREAT — Possible gunshot sound detected "
                "(SIMULATED). Confidence 86%.",
                "critical", payload={"audio_label": "gunshot_sim",
                                     "confidence": 0.86})
            st.rerun()
        if c2.button("🗣 Simulate: Scream / distress", use_container_width=True,
                     key="audio_sim_scream"):
            make_alert(
                "audio_threat",
                "AUDIO THREAT — Scream / distress sound detected "
                "(SIMULATED). Confidence 78%.",
                "high", payload={"audio_label": "scream_sim",
                                 "confidence": 0.78})
            st.rerun()
        st.caption("All audio alerts travel through the same pipeline: "
                   "Alert Log → Threat Score → Incident Report → Offline "
                   "Queue → SHA-256 Ledger.")