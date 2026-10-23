#!/usr/bin/env python3
"""BorderEye /brag-slim launch video generator — 20s, 1920x1080, 30fps."""
from PIL import Image, ImageDraw, ImageFont
import os, subprocess

OUT = r"C:\Users\Krish\OneDrive\Desktop\BorderEyev1.0\BorderEye\brag-output"
WORK = os.path.join(OUT, "work")
LOGO = r"C:\Users\Krish\OneDrive\Desktop\BorderEyev1.0\BorderEye\assets/logo.png"
FINAL = os.path.join(OUT, "brag.mp4")
POSTER = os.path.join(OUT, "brag.jpg")
COPY = os.path.join(OUT, "share-copy.txt")

W, H = 1920, 1080
FPS = 30
TOTAL = 20  # seconds

DARK   = (17, 24, 39)    # #111827
LIGHT  = (243, 241, 235) # #F3F1EB
INK    = (17, 24, 39)    # #111827
MUTED  = (75, 85, 99)    # #4B5563
ACCENT = (6, 182, 212)   # #06B6D4
SIH    = (13, 59, 102)   # #0D3B66

def hexrgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (255,)

font_big  = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 64)
font_md   = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
font_sm   = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
font_xsm  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)

logo = Image.open(LOGO).convert("RGBA")

def new(bg=LIGHT):
    return Image.new("RGB", (W, H), bg)

def paste_logo(img, size, y, alpha=255):
    l = logo.resize((size, size), Image.LANCZOS)
    if alpha < 255:
        l.putalpha(int(255 * alpha / 255))
    img.paste(l, ((W - size) // 2, y), l)
    return img

def text(img, s, font, color, x, y, anchor="mm"):
    d = ImageDraw.Draw(img)
    d.text((x, y), s, font=font, fill=color, anchor=anchor)
    return img

os.makedirs(WORK, exist_ok=True)

scenes = []

# ── Scene 1: Hook (2.5s) ──────────────────────────────────────
img = new(DARK)
paste_logo(img, 280, H//2 - 100)
text(img, "The border never sleeps — and neither do we.", font_md, (255, 255, 255), W//2, H//2 + 210)
scenes.append(("01_hook", img, 2.5))

# ── Scene 2: Reveal (3.5s) ────────────────────────────────────
img = new(LIGHT)
paste_logo(img, 200, 120)
text(img, "BorderEye", font_big, INK, W//2, 430)
text(img, "Border Intelligence & Video Analytics Engine", font_sm, MUTED, W//2, 500)
badges = [
    ("Python 3.14", "3776AB", 430, 640),
    ("Streamlit 1.58", "FF4B4B", 640, 640),
    ("YOLOv8", "FFC107", 850, 640),
    ("FastAPI", "009688", 1020, 640),
    ("167 tests passing", "4C1", 430, 690),
    ("SIH 2026", "0D3B66", 640, 690),
]
for label, hexc, bx, by in badges:
    d = ImageDraw.Draw(img)
    bw = d.textlength(label, font=font_xsm) + 32
    d.rounded_rectangle([bx, by - 16, bx + bw, by + 16], radius=8, fill=hexrgb(hexc))
    d.text((bx + bw//2, by), label, font=font_xsm, fill=(255, 255, 255), anchor="mm")
scenes.append(("02_reveal", img, 3.5))

# ── Scene 3: Highlights (7s) ──────────────────────────────────
img = new(LIGHT)
text(img, "Everything runs at the edge", font_big, INK, W//2, 300)
highlights = [
    ("Edge-first AI", "CPU-only · ~$250 per post · video never leaves border", 200, 450),
    ("Tamper-evident ledger", "SHA-256 hash chain · immutable audit trail", 660, 450),
    ("No new cameras · no cloud", "AI over existing CCTV · 200 bytes upstream", 1120, 450),
]
for title, sub, x, y in highlights:
    text(img, title, font_md, ACCENT, x, y)
    text(img, sub, font_xsm, MUTED, x, y + 55)
text(img, "YOLOv8 + ByteTrack + ANPR", font_md, ACCENT, W//2, 660)
scenes.append(("03_highlights", img, 7))

# ── Scene 4: Punchline (4s) ───────────────────────────────────
img = new(LIGHT)
text(img, "Smart India Hackathon 2026", font_big, SIH, W//2, 340)
text(img, "Problem Statement SIH26187", font_md, INK, W//2, 410)
text(img, "Every AI-CCTV platform assumes good bandwidth, good cameras", font_xsm, MUTED, W//2, 500)
text(img, "and infinite trust in every alert", font_xsm, MUTED, W//2, 540)
text(img, "Border posts have none of those three.", font_big, INK, W//2, 590)
scenes.append(("04_punchline", img, 4))

# ── Scene 5: Outro (3s) ───────────────────────────────────────
img = new(LIGHT)
text(img, "Watch the full demo", font_md, INK, W//2, 380)
text(img, "youtu.be/RPTl-EkL2XM", font_md, ACCENT, W//2, 440)
text(img, "YOLOv8 + ByteTrack + ANPR · SIH26187", font_xsm, MUTED, W//2, 500)
text(img, "Edge-first · Tamper-evident · CPU-only · SIH 2026", font_xsm, INK, W//2, 620)
scenes.append(("05_outro", img, 3))

# Save each scene as PNG sequence
scene_files = []
for name, img, dur in scenes:
    n_frames = int(dur * FPS)
    for i in range(n_frames):
        p = os.path.join(WORK, f"{name}_{i:03d}.png")
        img.save(p)
        scene_files.append(p)

# Concat with crossfade using ffmpeg
list_file = os.path.join(WORK, "concat.txt")
# Build filter: concatenate with fade transitions between scenes
inputs = []
filter_parts = []
ts = 0
for idx, (name, img, dur) in enumerate(scenes):
    n_frames = int(dur * FPS)
    inputs.extend(["-i", os.path.join(WORK, f"{name}_000.png")])
    # For each scene, all frames are the same image repeated
    # ffmpeg will handle single image input with -t duration

# Simpler approach: make per-scene short videos from single PNG, then concat
scene_videos = []
for idx, (name, img, dur) in enumerate(scenes):
    sv = os.path.join(WORK, f"{name}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", os.path.join(WORK, f"{name}_000.png"),
        "-c:v", "libx264", "-t", str(dur), "-pix_fmt", "yuv420p", "-r", str(FPS), sv
    ], capture_output=True)
    scene_videos.append(sv)

# Concat list for ffmpeg
concat_list = os.path.join(WORK, "scenes.txt")
with open(concat_list, "w") as f:
    for sv in scene_videos:
        f.write(f"file '{sv}'\n")

subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
    "-c", "copy", FINAL
], capture_output=True)

# Poster from frame 0
subprocess.run([
    "ffmpeg", "-y", "-ss", "6", "-i", FINAL, "-vframes", "1", "-q:v", "2", POSTER
], capture_output=True)

# Share copy
with open(COPY, "w", encoding="utf-8") as f:
    f.write("""BorderEye — AI border surveillance over existing CCTV. Edge-first, tamper-evident, CPU-only. No new cameras, no cloud, no GPU. Smart India Hackathon 2026 · SIH26187.
Watch the full demo: youtu.be/RPTl-EkL2XM""")

print(f"DONE: {FINAL}")
print(f"POSTER: {POSTER}")
print(f"COPY: {COPY}")