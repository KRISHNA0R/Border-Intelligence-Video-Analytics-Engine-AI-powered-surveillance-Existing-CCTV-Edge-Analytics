# /brag-slim Plan — BorderEye

**Input:** Project directory `C:\Users\Krish\OneDrive\Desktop\BorderEyev1.0\BorderEye` (local project, no URL input).
**Format:** landscape 1920×1080, 30 fps, ~20 s.
**Tone:** `default` — punchy, playful, clean.

---

## One-sentence

BorderEye is an AI layer over existing CCTV cameras at border outposts — it runs YOLOv8 + ByteTrack detection and ANPR locally at the edge, raises instant alerts, and proves every alert is genuine with a tamper-evident SHA-256 ledger. No new cameras, no cloud, no GPU.

**Who it's for:** Border outposts and checkposts with poor bandwidth, untrusted cameras, and zero GPU budget.
**What sets it apart:** Runs fully offline on CPU (~$250 post), video never leaves the post (~200 bytes metadata upstream), hash-chain makes the audit trail provably immutable.
**Most impressive claim:** "Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three." — SIH26187 problem statement.
**Visual hook:** The BorderEye shield logo + rotating doctrine slogan ("Vigilance is the price of every peaceful dawn.") above the login card.
**Real UI shown:** Login card with live rotating slogan, plus hero badge stack (Python 3.14 / Streamlit 1.58 / YOLOv8 / FastAPI / 167 tests passing / SIH 2026).
**Share caption:** BorderEye — AI border surveillance over existing CCTV. Edge-first, tamper-evident, CPU-only. SIH 2026 · SIH26187. Watch: youtu.be/RPTl-EkL2XM

---

## Storyboard (durations sum to ~20 s)

| Scene | Time | Content | Duration |
|---|---|---|---|
| 1 Hook | 0.0–2.5 s | Dark `#111827` bg. BorderEye shield logo fades in center. Tagline fades below: *"The border never sleeps — and neither do we."* | 2.5 s |
| 2 Reveal | 2.5–6.0 s | White cards slide/fade in. Title: *"BorderEye — Border Intelligence & Video Analytics Engine"*. Badge stack appears one-by-one: Python 3.14 · Streamlit 1.58 · YOLOv8 · FastAPI · 167 tests · SIH 2026. | 3.5 s |
| 3 Highlights | 6.0–13.0 s | Three sharp highlights, each ~2 s, appearing left-to-right with soft fade:<br/>(a) Edge-first AI — CPU-only, ~$250/post, video never leaves the post<br/>(b) Tamper-evident — SHA-256 hash-chain, provably immutable audit trail<br/>(c) No new cameras · no cloud — AI over existing CCTV | 7.0 s |
| 4 Punchline | 13.0–17.0 s | Big type: *"Smart India Hackathon 2026 · Problem Statement SIH26187"*. Sub: *"Problem statement says every AI-CCTV platform assumes good bandwidth, good cameras, infinite trust. Border posts have none of those three."* | 4.0 s |
| 5 Outro | 17.0–20.0 s | Center: YouTube play badge → `youtu.be/RPTl-EkL2XM` ("Full Working Demo: YOLOv8 + ByteTrack + ANPR"). Bottom: "BorderEye · Edge-first · Tamper-evident · CPU-only · SIH 2026". Fade to black at 20 s. | 3.0 s |

---

## Visual identity (from source)

- Brand bg: `#F3F1EB` (light warm) / login dark `#111827`
- Accent: `#06B6D4` (cyan)
- Cards: `#FFFFFF`, border `#D9D6CE`
- Text near-black: `#111827`
- Font: clean sans-serif (system: Inter/Segoe UI, fall back)

---

## Audio

Single generated backing track — low, warm synth pad with soft pulse, music-key consistent. No harsh SFX. Voiceover off (per default). Music mixed soft under any on-screen text.

---

## Notes

- No secrets, API keys, customer names, or emails carried into the video. The login card shows `admin@gmail.com` (a documented demo credential in the README) — swapped to fictional `admin@border.post` in the video per the secrets rule.
- Real UI reused: login card + rotating doctrine slogan + badge stack copied from `web_demo.py` / README, not re-created from scratch.
- Poster (`brag.jpg`) = strongest settled frame at ~6 s (badge stack fully in, not mid-transition), baked in as frame 0 of `brag.mp4`.
