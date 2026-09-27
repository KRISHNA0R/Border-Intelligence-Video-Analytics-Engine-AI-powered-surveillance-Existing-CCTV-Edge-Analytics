"""
BorderEye — Border Intelligence & Video Analytics Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full working dashboard with:
  • YOLOv8 / HOG person + vehicle detection
  • Virtual fence intrusion alerts
  • ANPR (number plate recognition)
  • Tamper-evident SHA-256 hash chain
  • Camera signal-loss detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import streamlit as st
import cv2
import numpy as np
import time
import json
import hashlib
import tempfile
import random
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ═══════════════════════════════════════════════════
# OPTIONAL FEATURE MODULES — each import is guarded so that a missing
# dependency degrades to a clean message instead of breaking the core app.
# ═══════════════════════════════════════════════════
try:
    from modules.face_intel import render_face_intelligence
except Exception:
    render_face_intelligence = None
try:
    from modules.border_map import render_border_map
except Exception:
    render_border_map = None
try:
    from modules.audio_threat import analyze_wav, render_audio_threat
except Exception:
    analyze_wav = None
    render_audio_threat = None

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(
    page_title="BorderEye — Border Intelligence & Video Analytics Engine",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════
# THEME — Light Command Center (design only)
# Inter for UI, JetBrains Mono for technical data.
# ═══════════════════════════════════════════════════

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, .stApp {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    background-color: #F3F1EB;
    color: #17202A;
}
/* Never override Streamlit's icon glyphs (aria-hidden spans carry the
   Material Symbols ligatures — forcing Inter there leaks raw text like
   "uploadUpload" / "expand_more" into buttons, expanders, popovers). */
.stApp [aria-hidden="true"] {
    font-family: 'Material Symbols Rounded' !important;
}
h1 { font-size: 1.4rem !important; font-weight: 700 !important; letter-spacing: -0.01em; color: #17202A; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #17202A; }
h3 { font-size: 0.98rem !important; font-weight: 600 !important; }
/* Command-center surfaces: white cards on warm-cream background */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #D9D6CE;
    border-radius: 10px; padding: 10px 14px;
    box-shadow: 0 1px 2px rgba(23,32,42,0.05);
}
[data-testid="stExpander"] details { background: #FFFFFF; border: 1px solid #D9D6CE; border-radius: 10px !important; }
[data-testid="stImage"] img { border-radius: 8px; border: 1px solid #D9D6CE; }
.stButton > button { border-radius: 8px !important; font-weight: 600 !important; }
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, .stTimeInput input {
    border-radius: 8px !important;
}
div[data-baseweb="select"] > div { border-radius: 8px !important; }
.stAlert { border-radius: 10px !important; }
button[data-baseweb="tab"] { border-radius: 8px 8px 0 0 !important; }
div[data-testid="stFileUploader"] section { border-radius: 8px !important; border: 1px dashed #D8D4C9 !important; }
hr { border-color: #D9D6CE !important; }
code, kbd, .tech, .mono {
    font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace !important;
}
.tech { font-size: 0.78rem; letter-spacing: 0.01em; color: #667085; }
/* Restrained security colors — green = normal, amber = warning,
   red = critical, blue = accent only */
.sev-critical { color: #DC2626; font-weight: 600; }
.sev-high { color: #D97706; font-weight: 600; }
.sev-warning { color: #B45309; font-weight: 600; }
.sev-safe { color: #16A34A; font-weight: 600; }
/* Status pills used across the command center */
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; }
.pill-live { background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
.pill-demo { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.pill-off  { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
.pill-info { background: #DBEAFE; color: #1D4ED8; border: 1px solid #BFDBFE; }
/* Camera card */
.cam-card { background: #FFFFFF; border: 1px solid #D9D6CE; border-radius: 12px;
            padding: 12px 14px; box-shadow: 0 1px 3px rgba(23,32,42,0.06); }
.cam-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.cam-id { font-weight: 700; font-size: 0.92rem; color: #17202A; }
.cam-loc { font-size: 0.74rem; color: #667085; }
/* Map success legend etc */
.map-note { font-size: 0.75rem; color: #667085; background: #FFFFFF;
            border: 1px solid #D9D6CE; border-radius: 8px; padding: 8px 12px; }
/* ──────────────────────────────────────────────────
   HYBRID TACTICAL SYSTEM — 70% light + 30% dark
   ────────────────────────────────────────────────── */
/* Dark sidebar — the primary tactical surface */
[data-testid="stSidebar"] { background: #111827; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #D1D5DB; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F3F4F6 !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #1F2937; border-color: #374151;
    color: #F3F4F6;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #F3F4F6 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1F2937; color: #F3F4F6; border: 1px solid #374151;
}
[data-testid="stSidebar"] .stButton > button:hover { border-color: #06B6D4; }
/* Compact tactical navigation — active item gets a cyan edge */
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 2px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 10px; margin: 1px 0; border-radius: 6px;
    border-left: 3px solid transparent; cursor: pointer;
    font-size: 0.85rem; color: #D1D5DB;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #1F2937; border-left: 3px solid #06B6D4; color: #E5E7EB;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color: #06B6D4; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #1F2937; border-color: #374151 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section * { color: #D1D5DB !important; }
/* Compact status chips */
.chip { display: inline-block; padding: 1px 8px; border-radius: 999px;
        font-size: 0.66rem; font-weight: 700; letter-spacing: 0.06em;
        vertical-align: middle; }
.chip-live   { background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
.chip-sim    { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.chip-off    { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
.chip-synced { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.chip-queued { background: #FEF9C3; color: #854D0E; border: 1px solid #FDE047; }
.chip-warn   { background: #FFF7ED; color: #C2410C; border: 1px solid #FED7AA; }
.chip-crit   { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
.chip-info   { background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
/* Tactical KPI cards */
.kpi { background: #FFFFFF; border: 1px solid #D9D6CE; border-radius: 10px;
       padding: 10px 12px; box-shadow: 0 1px 2px rgba(23,32,42,0.05); }
.kpi .kpi-label { font-size: 0.62rem; letter-spacing: 0.1em; color: #667085;
                  font-weight: 700; text-transform: uppercase; }
.kpi .kpi-val { font-size: 1.45rem; font-weight: 700; color: #17202A;
                font-family: 'JetBrains Mono', Consolas, monospace; line-height: 1.25; }
.kpi .kpi-state { font-size: 0.66rem; font-weight: 600; color: #16A34A; }
.kpi .kpi-state.warn { color: #D97706; }
.kpi .kpi-state.crit { color: #DC2626; }
/* Camera feed frame header */
.feed-head { display: flex; align-items: center; justify-content: space-between;
             background: #1F2937; color: #F3F4F6; border-radius: 10px 10px 0 0;
             padding: 8px 14px; font-size: 0.78rem; }
.feed-head .fh-id { font-weight: 700; letter-spacing: 0.06em; color: #F3F4F6; }
.feed-head .fh-loc { color: #9CA3AF; font-size: 0.7rem; }
.feed-head .fh-tag { color: #22D3EE; font-size: 0.66rem; font-weight: 700;
                     letter-spacing: 0.1em; }
/* Security event console */
.acard { background: #FFFFFF; border: 1px solid #D9D6CE; border-left: 4px solid #16A34A;
         border-radius: 8px; padding: 10px 12px; margin: 6px 0;
         box-shadow: 0 1px 2px rgba(23,32,42,0.04); }
.acard.crit { border-left-color: #DC2626; }
.acard.high { border-left-color: #D97706; }
.acard.med  { border-left-color: #D97706; }
.acard.low  { border-left-color: #16A34A; }
.acard .ac-title { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em;
                   color: #17202A; text-transform: uppercase; }
.acard .ac-meta { font-size: 0.7rem; color: #667085;
                  font-family: 'JetBrains Mono', Consolas, monospace; margin-top: 2px; }
.acard .ac-risk { float: right; font-size: 0.72rem; font-weight: 800; }
.acard .ac-expl { font-size: 0.78rem; color: #374151; margin-top: 4px; }
/* Module panels */
.panel { background: #FFFFFF; border: 1px solid #D9D6CE; border-radius: 12px;
         padding: 14px 16px; box-shadow: 0 1px 3px rgba(23,32,42,0.06); margin: 8px 0; }
.panel-title { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
               color: #667085; text-transform: uppercase; margin-bottom: 6px; }
.panel-ok { border-left: 4px solid #16A34A; }
.panel-critical { border-left: 4px solid #DC2626; }
/* Sidebar system footer */
.sys-foot { background: #1F2937; border: 1px solid #374151; border-radius: 10px;
            padding: 10px 12px; margin-top: 8px; }
.sys-foot .sf-title { font-size: 0.62rem; letter-spacing: 0.12em; color: #9CA3AF;
                      font-weight: 700; }
.sys-foot .sf-row { font-size: 0.74rem; color: #F3F4F6; margin-top: 6px;
                    font-family: 'JetBrains Mono', Consolas, monospace; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
       margin-right: 6px; vertical-align: middle; }
.dot-ok { background: #22C55E; box-shadow: 0 0 6px rgba(34,197,94,0.8); }
.dot-warn { background: #F59E0B; }
.dot-err { background: #EF4444; box-shadow: 0 0 6px rgba(239,68,68,0.8); }
/* Camera wall (2x2 security grid) */
.cam-card { background: #FFFFFF; border: 1px solid #D9D6CE;
            border-top: 3px solid #1F2937; border-radius: 10px;
            padding: 12px 14px; box-shadow: 0 1px 3px rgba(23,32,42,0.06); }
.cam-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.cam-id { font-weight: 700; font-size: 0.92rem; color: #17202A; }
.cam-loc { font-size: 0.74rem; color: #667085; }
/* Command-center header */
.cc-head { background: #FFFFFF; border: 1px solid #D9D6CE; border-radius: 12px;
           padding: 12px 16px; box-shadow: 0 1px 3px rgba(23,32,42,0.06);
           display: flex; align-items: center; justify-content: space-between; }
.cc-head .cc-title { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em;
                     color: #17202A; }
.cc-head .cc-sub { font-size: 0.72rem; color: #667085; margin-top: 2px; }
.cc-stats { display: flex; gap: 10px; align-items: center; }
.cc-stat { background: #F3F1EB; border: 1px solid #D9D6CE; border-radius: 8px;
           padding: 4px 10px; font-size: 0.68rem; color: #17202A; white-space: nowrap; }
.cc-stat b { font-family: 'JetBrains Mono', Consolas, monospace; }
/* ──────────────────────────────────────────────────
   HI-VISIBILITY TEXT — ALL MAIN CONTENT BOLD + NEAR-BLACK
   Scoped to the main block so the dark sidebar keeps its
   white-on-dark scheme untouched.
   ────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] { color: #0B0F19; }
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4,
[data-testid="stMainBlockContainer"] h5,
[data-testid="stMainBlockContainer"] h6,
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] label,
[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stMetricLabel"],
[data-testid="stMainBlockContainer"] [data-testid="stMetricValue"],
[data-testid="stMainBlockContainer"] .stMarkdown,
[data-testid="stMainBlockContainer"] .stText,
[data-testid="stMainBlockContainer"] .stAlert,
[data-testid="stMainBlockContainer"] code,
[data-testid="stMainBlockContainer"] pre {
    color: #0B0F19 !important;
    font-weight: 700 !important;
}
[data-testid="stMainBlockContainer"] input,
[data-testid="stMainBlockContainer"] textarea {
    color: #0B0F19 !important;
    font-weight: 600 !important;
}
/* Muted helper classes inside the main block go dark + bold too */
.tech { color: #0B0F19 !important; font-weight: 600 !important; }
.map-note { color: #0B0F19 !important; font-weight: 600 !important; }
.cam-loc { color: #111827 !important; font-weight: 600 !important; }
.acard .ac-meta { color: #111827 !important; font-weight: 600 !important; }
.kpi .kpi-label { color: #111827 !important; }
.cc-head .cc-sub { color: #111827 !important; font-weight: 600 !important; }
.panel-title { color: #0B0F19 !important; font-weight: 800 !important; }
.cc-stat { color: #0B0F19 !important; font-weight: 700 !important; }
.kpi .kpi-val { color: #0B0F19 !important; }
.cam-id { color: #0B0F19 !important; }
.acard .ac-title { color: #0B0F19 !important; }
/* ──────────────────────────────────────────────────
   HI-VISIBILITY MOTION — pulse, REC blink, hover-lift,
   ticker, scrollbar, selection, focus rings
   ────────────────────────────────────────────────── */
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }
    70%  { box-shadow: 0 0 0 9px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
@keyframes blink-rec {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.12; }
}
@keyframes ticker-scroll {
    0%   { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
.live-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
            background: #22C55E; animation: pulse-ring 1.8s ease-out infinite;
            box-shadow: 0 0 6px rgba(34,197,94,0.8);
            margin-right: 6px; vertical-align: middle; }
.rec-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
           background: #EF4444; animation: blink-rec 1.1s step-end infinite;
           box-shadow: 0 0 8px rgba(239,68,68,0.9);
           margin-right: 6px; vertical-align: middle; }
.rec-tag { color: #DC2626; font-weight: 800; letter-spacing: 0.1em;
           font-size: 0.68rem; }
[data-testid="stSidebar"] .dot-ok,
.cc-head .dot-ok { animation: pulse-ring 1.8s ease-out infinite; }
/* Custom tactical scrollbar */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #EAE7DF; }
::-webkit-scrollbar-thumb { background: #B7B2A6; border-radius: 6px;
                            border: 2px solid #F3F1EB; }
::-webkit-scrollbar-thumb:hover { background: #06B6D4; }
[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: #111827; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: #374151;
                                                      border: 2px solid #111827; }
/* Selection + focus visibility */
::selection { background: #06B6D4; color: #FFFFFF; }
.stButton > button:focus-visible,
.stTextInput input:focus-visible,
[data-baseweb="input"]:focus-within,
textarea:focus-visible {
    outline: 2px solid #06B6D4 !important;
    outline-offset: 1px;
}
/* Card hover-lift */
.kpi, .cam-card, .acard, .panel, .cc-head,
[data-testid="stMetric"] {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi:hover, .cam-card:hover, .acard:hover, .panel:hover, .cc-head:hover,
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(23,32,42,0.13);
}
/* Scrolling event ticker */
.ticker-wrap { overflow: hidden; background: #111827; border-radius: 8px;
               padding: 7px 12px; }
.ticker-wrap::before { content: "⚡ LIVE FEED"; color: #22D3EE;
                       font-weight: 800; font-size: 0.62rem;
                       letter-spacing: 0.12em; padding-right: 12px; }
.ticker { display: inline-flex; gap: 30px; white-space: nowrap;
          animation: ticker-scroll 28s linear infinite;
          font-size: 0.78rem; font-weight: 600; color: #E5E7EB; }
.ticker-wrap:hover .ticker { animation-play-state: paused; }
.tk-crit { color: #F87171; } .tk-high { color: #FBBF24; }
.tk-med  { color: #FDE68A; } .tk-low  { color: #A7F3D0; }
.tk-tag  { color: #9CA3AF; font-size: 0.68rem;
           font-family: 'JetBrains Mono', Consolas, monospace; }
/* ACK chips */
.ack-chip { display: inline-block; background: #ECFDF5; color: #047857;
            border: 1px solid #A7F3D0; border-radius: 999px;
            font-size: 0.62rem; font-weight: 800; letter-spacing: 0.06em;
            padding: 1px 8px; margin-right: 6px; }
.ack-chip-new { background: #FEF9C3; color: #854D0E; border-color: #FDE047; }
/* Health grid */
.health-grid { display: grid; gap: 8px;
               grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
.hcell { background: #F3F1EB; border: 1px solid #D9D6CE; border-radius: 8px;
         padding: 6px 10px; }
.hcell .hc-tag { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em;
                 color: #667085; text-transform: uppercase; }
.hcell .hc-val { font-size: 0.92rem; font-weight: 700; color: #0B0F19;
                 font-family: 'JetBrains Mono', Consolas, monospace; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════

# Weights fine-tuned on IDD (7 classes incl. autorickshaw). Versioned in
# the repo, so the hosted demo gets them too.
FINE_TUNED_DETECTOR = Path(__file__).parent / "models" / "weights" / "detection.pt"


@st.cache_resource
def load_yolo():
    """Load the fine-tuned detector if present, else stock YOLOv8n.

    Returns (model, is_fine_tuned) — the caller needs to know which, because
    the two have different class taxonomies and want different confidence
    cutoffs.
    """
    from ultralytics import YOLO
    if FINE_TUNED_DETECTOR.exists():
        return YOLO(str(FINE_TUNED_DETECTOR)), True
    return YOLO("yolov8n.pt"), False

@st.cache_resource
def load_hog():
    """Load OpenCV's HOG person detector as fallback.

    Returns None on OpenCV 5+, which removed HOGDescriptor entirely. The
    hosted demo has no torch, so this fallback is its *primary* path — a
    hard failure here takes the whole public page down, which is why it
    degrades instead of raising.
    """
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


@st.cache_resource
def load_motion_detector():
    """Last-resort detector: background subtraction, available in every
    OpenCV version. Detects movement, not people — labelled as such."""
    return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32,
                                              detectShadows=False)

@st.cache_resource
def load_ocr():
    """Load EasyOCR for ANPR."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

# Detection engine, best available first: YOLO -> HOG -> motion.
USE_YOLO = False
hog = None
motion = None
try:
    model, FINE_TUNED = load_yolo()
    USE_YOLO = True
    ENGINE = "YOLOv8 (IDD fine-tuned)" if FINE_TUNED else "YOLOv8 (stock COCO)"
except Exception:
    model = None
    FINE_TUNED = False
    hog = load_hog()
    if hog is not None:
        ENGINE = "HOG"
    else:
        motion = load_motion_detector()
        ENGINE = "Motion"


@st.cache_resource
def load_face_detector():
    """Haar face detector — ships inside OpenCV, no downloads, no torch."""
    try:
        clf = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if clf.empty() else clf
    except Exception:
        return None


face_clf = load_face_detector()

# EasyOCR is built on first use, NOT at import. Constructing the Reader
# downloads its detection+recognition models and costs ~700MB of RSS; doing
# that at module scope spends the hosted tier's whole memory budget and its
# startup window before the first frame renders. load_ocr() is
# @st.cache_resource, so the first caller pays once and the rest are free.

# ═══════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════

if "alerts" not in st.session_state:
    st.session_state.alerts = []
# Test/CI hook: BORDEREYE_TEST_AUTOLOGIN=1 skips the login wall so
# AppTest-style harnesses can reach every section without widget-state
# churn from the auth gate's st.stop(). Harmless in production.
if os.environ.get("BORDEREYE_TEST_AUTOLOGIN") == "1":
    st.session_state.authed = True
    st.session_state.login_user = "admin@gmail.com"
if "prev_hash" not in st.session_state:
    st.session_state.prev_hash = "0" * 64
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "camera_status" not in st.session_state:
    st.session_state.camera_status = {
        "CAM-01": True, "CAM-02": True, "CAM-03": True
    }
if "plate_cache" not in st.session_state:
    st.session_state.plate_cache = {}
if "nav_section" not in st.session_state:
    st.session_state.nav_section = "🏠 Command Center"
if "plate_log" not in st.session_state:
    st.session_state.plate_log = []
if "_ocr_cache" not in st.session_state:
    st.session_state._ocr_cache = {}
if "map_coords" not in st.session_state:
    _mc = {}
    try:
        _sites = Path("data/sites.json")
        if _sites.exists():
            _loaded = json.loads(_sites.read_text(encoding="utf-8"))
            if isinstance(_loaded, dict):
                _mc = {k: v for k, v in _loaded.items() if isinstance(v, dict)}
    except Exception:
        _mc = {}
    st.session_state.map_coords = _mc
if "_seen_alerts" not in st.session_state:
    st.session_state._seen_alerts = set()
if "_acked" not in st.session_state:
    st.session_state._acked = set()
if "_started" not in st.session_state:
    st.session_state._started = time.time()

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

W, H = 640, 480
FENCE = np.array([[180, 120], [460, 120], [460, 380], [180, 380]], np.int32)

# Class indices are taxonomy-specific: stock COCO puts bus at 5 and truck
# at 7, the fine-tuned model packs its seven classes into 0-6. Reading the
# map off the loaded model is what stops an autorickshaw being reported as
# a truck.
_WANTED = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "autorickshaw"}
_VEHICLES = {"car", "motorcycle", "bus", "truck", "autorickshaw"}

if USE_YOLO and getattr(model, "names", None):
    _names = {int(i): str(n) for i, n in dict(model.names).items()}
    TARGET_CLASSES = {i: n for i, n in _names.items() if n in _WANTED}
    VEHICLE_CLASSES = {i for i, n in TARGET_CLASSES.items() if n in _VEHICLES}
else:
    TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                      5: "bus", 7: "truck"}
    VEHICLE_CLASSES = {2, 3, 5, 7}

# 0.45 suits stock COCO; the fine-tuned model peaks at 0.25 (F1), and a
# border demo should favour recall over precision.
DETECT_CONF = 0.25 if (USE_YOLO and FINE_TUNED) else 0.45

INDIAN_PLATE_PATTERN = (
    r"(?:^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$)"
)

# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

def compute_risk_score(event_type, severity, confidence, object_class=None, in_fence=False):
    """Deterministic 0-100 risk score from real detection signals.

    This is a documented weighted formula, NOT a trained ML model:
      severity base + detection confidence + fence breach + night hours.
    Returns (score, level, reasons) — reasons explain every point.
    """
    base = {"critical": 70, "high": 55, "medium": 35, "low": 15}.get(severity, 25)
    reasons = [f"Severity '{severity}' (base {base})"]
    score = float(base)

    conf = float(confidence or 0)
    conf_pts = round(conf * 20, 1)
    score += conf_pts
    reasons.append(f"Detection confidence {conf:.0%} (+{conf_pts:g})")

    if in_fence and (object_class == "person" or event_type == "fence_intrusion"):
        score += 10
        reasons.append("Person inside virtual fence (+10)")

    hour = datetime.now().hour
    if hour >= 22 or hour < 5:
        score += 5
        reasons.append(f"Night hours ({hour:02d}:00) (+5)")

    if event_type == "anpr_match":
        score += 5
        reasons.append("Plate on watchlist (+5)")

    score = int(min(99, max(1, round(score))))
    if score >= 80:
        level = "CRITICAL"
    elif score >= 60:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level, reasons


def create_alert(event_type, explanation, severity, payload=None, confidence=None,
                 object_class=None, in_fence=False):
    """Create alert with hash chain + deterministic risk score."""
    if confidence is None:
        # System events (not model detections) get fixed honest values.
        confidence = {"signal_loss": 0.99, "signal_restored": 0.99}.get(event_type, 0.80)
    risk_score, risk_level, risk_reasons = compute_risk_score(
        event_type, severity, confidence, object_class, in_fence)
    alert = {
        "event_id": f"e{uuid4_hex()}",
        "prev_hash": st.session_state.prev_hash,
        "timestamp": datetime.now().isoformat(),
        "site_id": "BOP-01",
        "camera_id": "CAM-01",
        "event_type": event_type,
        "severity": severity,
        "confidence": round(float(confidence), 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "explanation": explanation,
        "simulated": bool(str(explanation).startswith("SIMULATED")
                          or "(SIMULATED)" in str(explanation)),
        "payload": payload or {},
    }
    # Compute hash
    payload_for_hash = {k: v for k, v in alert.items() if k not in ("prev_hash", "hash")}
    h = hashlib.sha256(json.dumps(payload_for_hash, sort_keys=True, default=str).encode()).hexdigest()
    alert["hash"] = h
    alert["prev_hash"] = st.session_state.prev_hash
    st.session_state.prev_hash = h
    st.session_state.alerts.append(alert)
    # Low-bandwidth layer: metadata only travels; video never leaves edge.
    alert["link_status"] = outbox_push(alert)
    return alert


def uuid4_hex():
    """Generate a random hex string (no uuid import needed)."""
    import random
    return ''.join(random.choices('0123456789abcdef', k=16))


def verify_chain():
    """Verify hash chain integrity."""
    chain = st.session_state.alerts
    if len(chain) < 2:
        return True
    for i in range(1, len(chain)):
        prev = chain[i - 1].copy()
        prev.pop("hash", None)
        expected = hashlib.sha256(
            json.dumps(prev, sort_keys=True, default=str).encode()
        ).hexdigest()
        if chain[i].get("prev_hash") != chain[i - 1].get("hash"):
            return False
    return True


def is_in_fence(point):
    """Check if a point is inside the virtual fence."""
    return cv2.pointPolygonTest(FENCE, (float(point[0]), float(point[1])), False) >= 0


def build_alert_pdf():
    """Compile every alert this session into an English PDF report.

    Returns PDF bytes. Pure-python (fpdf2), works offline on the edge device.
    """
    from fpdf import FPDF

    def _t(s):
        return str(s).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BorderEye - Instant Alert Report",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _t(
        f"Site: BOP-01 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"| Operator: {st.session_state.get('login_user', 'operator')}"),
        new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _t(
        f"Total alerts: {len(st.session_state.alerts)} | "
        f"Hash chain: {'VALID' if verify_chain() else 'BROKEN'}"),
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    if not st.session_state.alerts:
        pdf.cell(0, 8, "No alerts recorded in this session.",
                 new_x="LMARGIN", new_y="NEXT")
    for a in st.session_state.alerts:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _t(
            f"[{a['severity'].upper()}] {a['event_type'].upper()}  "
            f"(Risk {a.get('risk_score', '-')}/100 {a.get('risk_level', '')})"),
            new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _t(a.get("explanation", "")))
        pdf.cell(0, 6, _t(
            f"Time: {a['timestamp'][:19]} | ID: {a['event_id']} | "
            f"Link: {a.get('link_status', 'synced')}"),
            new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    return bytes(pdf.output())


def _orient(a, b, c):
    """Cross-product orientation of triplet (a,b,c)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segs_cross(a1, a2, b1, b2):
    """True if segment a1-a2 intersects segment b1-b2 (tripwire crossing)."""
    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)
    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


OUTBOX_PATH = Path(__file__).parent / "data" / "outbox.jsonl"


def link_is_up():
    """Low-bandwidth link state. OFF = edge keeps working, events queue."""
    return bool(st.session_state.get("link_up", True))


def _outbox_save():
    """Persist outbox statuses to JSONL (survives restart)."""
    try:
        OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTBOX_PATH, "w", encoding="utf-8") as f:
            for item in st.session_state.get("outbox", []):
                f.write(json.dumps(item) + "\n")
    except Exception:
        pass


def outbox_push(alert):
    """Route an alert through the low-bandwidth layer.

    Link UP   -> metadata marked 'synced' (this is all that travels).
    Link DOWN -> metadata marked 'queued', video stays on the edge device.
    """
    status = "synced" if link_is_up() else "queued"
    st.session_state.setdefault("outbox", []).append({
        "event_id": alert["event_id"], "status": status,
        "timestamp": alert["timestamp"], "event_type": alert["event_type"],
    })
    st.session_state.setdefault("outbox_status", {})[alert["event_id"]] = status
    _outbox_save()
    return status


def outbox_flush():
    """Push all queued events when the link restores. Returns count synced."""
    n = 0
    for item in st.session_state.get("outbox", []):
        if item["status"] == "queued":
            item["status"] = "synced"
            st.session_state.setdefault("outbox_status", {})[item["event_id"]] = "synced"
            n += 1
    _outbox_save()
    return n


ANCHOR_PATH = Path(__file__).parent / "data" / "anchors.jsonl"


def anchor_now():
    """Checkpoint the day's edge-ledger head-hash (blockchain anchor, demo grade).

    Writes a self-chained anchor record locally. If a timestamp-service URL
    is configured (e.g. a Polygon Amoy relayer), the anchor is POSTed there
    and the receipt stored as public proof.
    """
    anchors = st.session_state.get("anchors", [])
    prev = anchors[-1]["anchor_hash"] if anchors else "GENESIS"
    rec = {
        "anchor_id": f"a{uuid4_hex()[:12]}",
        "head_hash": st.session_state.prev_hash,
        "events": len(st.session_state.alerts),
        "timestamp": datetime.now().isoformat(),
        "operator": st.session_state.get("login_user", "operator"),
        "prev_anchor": prev,
    }
    rec["anchor_hash"] = hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode()).hexdigest()
    url = (st.session_state.get("anchor_url") or "").strip()
    if url:
        try:
            import requests
            r = requests.post(url, json=rec, timeout=8)
            rec["receipt"] = f"HTTP {r.status_code}"
        except Exception as e:
            rec["receipt"] = f"failed ({e.__class__.__name__})"
    else:
        rec["receipt"] = "local checkpoint (no timestamp URL set)"
    anchors.append(rec)
    st.session_state.anchors = anchors
    try:
        ANCHOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ANCHOR_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec


def verify_anchors():
    """Recompute every anchor hash + link. True if the anchor log is intact."""
    anchors = st.session_state.get("anchors", [])
    for i, a in enumerate(anchors):
        core = {k: v for k, v in a.items()
                if k not in ("anchor_hash", "receipt")}
        if hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest() != a.get("anchor_hash"):
            return False
        want = "GENESIS" if i == 0 else anchors[i - 1]["anchor_hash"]
        if a.get("prev_anchor") != want:
            return False
    return True


def groq_summarize(alerts, api_key):
    """Summarize alert metadata with Groq LLM.

    Returns 2-line English summary + 1 Hindi action line. Demo/HQ mode only:
    alert metadata leaves the device for this call.
    """
    import requests
    compact = [{
        "type": a["event_type"], "sev": a["severity"],
        "risk": a.get("risk_score"), "time": a["timestamp"][:19],
        "track": (a.get("payload") or {}).get("track_id"),
        "note": a.get("explanation", "")[:200],
    } for a in alerts]
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system",
                 "content": "You are a border-security duty assistant. Given "
                            "BorderEye alert metadata JSON, reply in exactly 3 "
                            "short lines: lines 1-2 form a 2-line English "
                            "incident summary; line 3 is one Hindi action line "
                            "starting with 'कार्रवाई:'. No extra text."},
                {"role": "user", "content": json.dumps(compact)},
            ],
            "temperature": 0.3, "max_tokens": 250,
        },
        timeout=25,
    )
    if r.status_code != 200:
        return f"Groq error HTTP {r.status_code}: {r.text[:200]}"
    return r.json()["choices"][0]["message"]["content"]


def update_tracks(detections):
    """Track-state engine: tripwire crossing + loitering + fast movement.

    Rule-based heuristics over persistent YOLO track IDs — NOT an ML model.
    Only tracks with a valid track_id (>= 0) participate.
    """
    import math
    now = time.time()
    th = st.session_state.get("track_hist", {})
    seen = set()
    for det in detections:
        tid = det.get("track_id", -1)
        if tid is None or tid < 0:
            continue
        seen.add(tid)
        c = det["center"]
        st_ = th.get(tid)
        if st_ is None:
            th[tid] = {"prev": c, "first_seen": now, "loiter_fired": False,
                       "last_fast": 0.0, "last_cross": 0.0}
            continue
        prev = st_["prev"]
        # ── Tripwire crossing: movement vector vs every pen-drawn segment ──
        if prev != c:
            for i, seg in enumerate(st.session_state.get("tripwires", [])):
                (x1, y1), (x2, y2) = seg
                if segs_cross(prev, c, (x1, y1), (x2, y2)):
                    if now - st_["last_cross"] > 30:
                        st_["last_cross"] = now
                        create_alert(
                            "tripwire_crossing",
                            f"Track #{tid} ({det['class']}) crossed tripwire-{i + 1}.",
                            "high",
                            payload={"track_id": tid, "tripwire": i + 1},
                            confidence=det["confidence"],
                            object_class=det["class"],
                        )
            # ── Fast movement: large per-frame centroid jump ──
            d = math.hypot(c[0] - prev[0], c[1] - prev[1])
            if d > 45 and now - st_["last_fast"] > 60:
                st_["last_fast"] = now
                create_alert(
                    "fast_movement",
                    f"Track #{tid} moving unusually fast (~{d:.0f}px/frame).",
                    "medium",
                    payload={"track_id": tid, "px_per_frame": round(d, 1)},
                    confidence=det["confidence"],
                    object_class=det["class"],
                )
        # ── Loitering: same ID inside fence > 15 s ──
        if det["in_fence"]:
            if not st_["loiter_fired"] and now - st_["first_seen"] > 15:
                st_["loiter_fired"] = True
                create_alert(
                    "loitering",
                    f"Track #{tid} loitering inside fence for "
                    f"{now - st_['first_seen']:.0f}s.",
                    "medium",
                    payload={"track_id": tid,
                             "dwell_s": round(now - st_["first_seen"], 1)},
                    confidence=det["confidence"],
                    object_class=det["class"],
                )
        else:
            st_["first_seen"] = now
            st_["loiter_fired"] = False
        st_["prev"] = c
    # Prune stale tracks (keep memory bounded)
    for tid in list(th.keys()):
        if tid not in seen:
            del th[tid]
        if len(th) <= 80:
            break
    st.session_state.track_hist = th


def enhance_night(frame):
    """Brighten a dark frame with CLAHE on the LAB L-channel. Cheap, no models."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def is_night_hours():
    """Auto night window 19:00–06:00 local time."""
    h = datetime.now().hour
    return (h >= 19 or h < 6)


def night_active():
    """Manual toggle OR automatic night hours."""
    return bool(st.session_state.get("night_mode")) or is_night_hours()


def detect_frame(frame):
    """Run detection on a frame. Returns annotated frame + detections."""
    if night_active():
        frame = enhance_night(frame)
    frame = cv2.resize(frame, (W, H))
    detections = []

    if USE_YOLO and model is not None:
        # Built-in ByteTrack: persistent IDs across frames, no extra deps.
        results = model.track(frame, conf=DETECT_CONF, persist=True,
                              verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in TARGET_CLASSES:
                    continue
                conf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                in_fence = is_in_fence((cx, cy))
                cls_name = TARGET_CLASSES[cls]
                detections.append({
                    "class": cls_name, "class_id": cls,
                    "confidence": conf, "track_id": tid,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy), "in_fence": in_fence,
                })
    elif hog is not None:
        # HOG fallback — OpenCV 4.x only
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                               padding=(4, 4), scale=1.05)
        for (x, y, w, h), conf in zip(boxes, weights):
            cx, cy = x + w // 2, y + h // 2
            in_fence = is_in_fence((cx, cy))
            detections.append({
                "class": "person", "class_id": 0,
                "confidence": float(conf[0]),
                "bbox": (x, y, x + w, y + h),
                "center": (cx, cy), "in_fence": in_fence,
            })
    elif motion is not None:
        # Motion fallback — movement, not classification. Confidence is a
        # blob-area heuristic, not a model score; do not present it as one.
        mask = motion.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            area = cv2.contourArea(contour)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            in_fence = is_in_fence((cx, cy))
            detections.append({
                "class": "motion", "class_id": 0,
                "confidence": min(0.99, area / (W * H) * 8),
                "bbox": (x, y, x + w, y + h),
                "center": (cx, cy), "in_fence": in_fence,
            })

    # Draw
    vis = frame.copy()
    cv2.polylines(vis, [FENCE], True, (0, 255, 0), 2)
    cv2.putText(vis, "VIRTUAL FENCE", (210, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Pen-drawn tripwires (magenta)
    for i, seg in enumerate(st.session_state.get("tripwires", [])):
        (x1, y1), (x2, y2) = seg
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cv2.putText(vis, f"TW-{i + 1}", (int(x1) + 4, int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    persons = 0
    vehicles = 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["in_fence"] else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tid = det.get("track_id", -1)
        tag = f" #{tid}" if tid is not None and tid >= 0 else ""
        label = f"{det['class']}{tag} {det['confidence']:.0%}"
        cv2.putText(vis, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if det["class"] == "person":
            persons += 1
            if det["in_fence"]:
                # Throttle per track (60 s) so the log doesn't flood per frame.
                fl = st.session_state.get("fence_last", {})
                if tid is None or tid < 0 or time.time() - fl.get(tid, 0) > 60:
                    if tid is not None and tid >= 0:
                        fl[tid] = time.time()
                        st.session_state.fence_last = fl
                    create_alert(
                        "fence_intrusion",
                        f"Person{tag} detected crossing virtual fence. "
                        f"Confidence: {det['confidence']:.0%}.",
                        "high",
                        payload={"track_id": tid},
                        confidence=det["confidence"],
                        object_class=det["class"],
                        in_fence=True,
                    )
        elif det["class_id"] in VEHICLE_CLASSES:
            vehicles += 1

    # ── Track engine: tripwire crossing + loitering + fast movement ──
    update_tracks(detections)

    # ── Crowd gathering: 5+ persons in one frame (rule-based, 1/min) ──
    if persons >= 5:
        last = st.session_state.get("crowd_last_alert", 0)
        if time.time() - last > 60:
            st.session_state.crowd_last_alert = time.time()
            create_alert(
                "crowd_gathering",
                f"Crowd gathering: {persons} persons in one frame.",
                "medium",
                payload={"person_count": persons},
                confidence=0.80,
            )

    # ── Face detection (Haar, built-in OpenCV) ──
    faces = 0
    if st.session_state.get("face_detect") and face_clf is not None:
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        for (fx, fy, fw, fh) in face_clf.detectMultiScale(gray, 1.1, 4):
            faces += 1
            cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            cv2.putText(vis, "FACE", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    st.session_state.last_faces = faces

    # ── Behavioral stats (every processed frame) ──
    st.session_state.stats_frames = st.session_state.get("stats_frames", 0) + 1
    tot = st.session_state.get("stats_total", {})
    for d in detections:
        tot[d["class"]] = tot.get(d["class"], 0) + 1
    if faces:
        tot["face"] = tot.get("face", 0) + faces
    st.session_state.stats_total = tot
    hr = datetime.now().strftime("%H:00")
    hh = st.session_state.get("stats_hourly", {})
    hh[hr] = hh.get(hr, 0) + len(detections) + faces
    st.session_state.stats_hourly = hh

    # ── Night-time movement alert (throttled to 1/min) ──
    if night_active() and (persons + vehicles) > 0:
        last = st.session_state.get("night_last_alert", 0)
        if time.time() - last > 60:
            st.session_state.night_last_alert = time.time()
            create_alert(
                "night_movement",
                f"Night-time movement: {persons} person(s), {vehicles} vehicle(s) in view.",
                "medium", confidence=0.80,
            )

    # HUD
    cv2.rectangle(vis, (0, 0), (W, 28), (20, 20, 30), -1)
    cv2.putText(vis, f"BorderEye | BOP-01 CAM-01 | Frame {st.session_state.frame_count} | "
                f"Persons: {persons} | Vehicles: {vehicles}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

    return vis, detections


def _plate_variants(crop):
    """Multiple preprocessed crop versions for OCR — upscale + enhance."""
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 11)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return [up,
            cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(cl.apply(g), cv2.COLOR_GRAY2BGR)]


def run_ocr_on_frame(frame):
    """Run OCR on frame for plate detection (multi-pass, honest output).

    Flow: plate region → crop → upscale/enhance → OCR → confidence.
    Returns list of plates: {text, confidence, bbox, unclear}.
    'unclear' is True when OCR confidence sits below the readable gate —
    the text is then reported as 'Plate text unclear', never invented.
    """
    reader = load_ocr()
    if reader is None:
        return []

    # Per-frame cache keyed by a tiny grayscale signature — avoids re-reading
    # the identical frame across Single-Frame inspections.
    cache = st.session_state.setdefault("_ocr_cache", {})
    _sig = hashlib.md5(
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120)
                   ).tobytes()).hexdigest()
    if _sig in cache:
        return cache[_sig]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / max(float(h), 1)
        area = w * h
        if not (2.0 < aspect < 6.0 and 1000 < area < 50000):
            continue
        crop = frame[max(0, y - 5):y + h + 5, max(0, x - 5):x + w + 5]
        if crop.size == 0:
            continue
        # Multi-pass OCR — the best reading across all crops wins.
        best = None
        for prep in _plate_variants(crop):
            try:
                res = reader.readtext(prep)
            except Exception:
                continue
            if not res:
                continue
            txt, cf = max(
                ((t.strip().upper(), float(c)) for (_, t, c) in res),
                key=lambda p: p[1], default=("", 0.0))
            if len(txt) >= 4 and (best is None or cf > best[1]):
                best = (txt, cf)
        if best:
            text, conf = best
            if conf > 0.5:
                plates.append({"text": text, "confidence": conf,
                               "bbox": (x, y, x + w, y + h), "unclear": False})
            else:
                plates.append({"text": "Plate text unclear", "confidence": conf,
                               "bbox": (x, y, x + w, y + h), "unclear": True})

    # Bound the per-frame cache so session memory stays flat.
    if len(cache) > 60:
        for k in list(cache)[:40]:
            del cache[k]
    cache[_sig] = plates

    # Append to the session plate log (shown in the ANPR section).
    if plates:
        for p in plates:
            st.session_state.setdefault("plate_log", []).append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": p["text"], "confidence": p["confidence"],
                "unclear": p.get("unclear", False), "cam": "CAM-01",
            })
        st.session_state.plate_log = st.session_state.plate_log[-60:]
    return plates


def generate_demo_frame(tick):
    """Generate a synthetic surveillance frame for demo."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Ground / grass
    frame[:] = (35, 50, 35)
    # Sky
    frame[:100] = (70, 50, 35)
    # Road
    cv2.rectangle(frame, (0, 340), (W, H), (65, 65, 65), -1)
    cv2.line(frame, (0, 370), (W, 370), (180, 180, 180), 1, cv2.LINE_AA)
    # Dashed centre line
    for dx in range(0, W, 40):
        cv2.line(frame, (dx, 370), (dx + 20, 370), (240, 240, 60), 2, cv2.LINE_AA)

    # ── Fence ──
    cv2.polylines(frame, [FENCE], True, (0, 220, 0), 2, cv2.LINE_AA)
    # Label top-centre of fence
    fc = FENCE.mean(axis=0).astype(int)
    cv2.putText(frame, "VIRTUAL FENCE", (fc[0] - 50, FENCE[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 1, cv2.LINE_AA)

    # ── Animated person ──
    px = 100 + (tick * 3) % 400
    py = 260 + int(8 * np.sin(tick * 0.25))
    # Body
    cv2.circle(frame, (px, py - 28), 11, (200, 200, 210), -1, cv2.LINE_AA)
    cv2.line(frame, (px, py - 16), (px, py + 10), (200, 200, 210), 2, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px - 14, py + 18), (200, 200, 210), 2, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px + 14, py + 18), (200, 200, 210), 2, cv2.LINE_AA)
    in_fence = is_in_fence((px, py))
    pcolor = (0, 0, 255) if in_fence else (0, 255, 255)
    cv2.rectangle(frame, (px - 18, py - 42), (px + 18, py + 24), pcolor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"PERSON 0.{random.randint(82,97)}", (px - 18, py - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, pcolor, 1, cv2.LINE_AA)

    # ── Animated car ──
    car_x = 500 - (tick * 4) % 650
    car_y = 380
    cv2.rectangle(frame, (car_x, car_y - 18), (car_x + 55, car_y + 8), (10, 80, 180), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (car_x + 8, car_y - 28), (car_x + 45, car_y - 18), (10, 60, 160), -1, cv2.LINE_AA)
    cv2.circle(frame, (car_x + 10, car_y + 10), 5, (30, 30, 30), -1)
    cv2.circle(frame, (car_x + 45, car_y + 10), 5, (30, 30, 30), -1)
    cv2.putText(frame, f"CAR 0.{random.randint(75,94)}", (car_x, car_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 165, 0), 1, cv2.LINE_AA)

    # ── HUD top bar ──
    cv2.rectangle(frame, (0, 0), (W, 26), (15, 15, 25), -1)
    cv2.putText(
        frame,
        f"BorderEye  |  BOP-01  CAM-01  |  Frame {tick}  |  {datetime.now().strftime('%H:%M:%S')}",
        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA,
    )
    # Detection count bar
    cv2.rectangle(frame, (0, H - 24), (W, H), (15, 15, 25), -1)
    cv2.putText(frame, f"Detection engine: {ENGINE}  |  Objects: 2",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    return frame, in_fence


def render_video_tab(video_path, key, loc=""):
    """One saved video: full Play (live analysis) or Single Frame inspect."""
    ph = st.empty()
    if loc:
        st.caption(f"📍 Location: {loc}")
    probe = cv2.VideoCapture(video_path)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()

    view = st.radio("View", ["▶ Play Video", "🖼 Single Frame"],
                    horizontal=True, key=f"view_{key}")

    if view == "🖼 Single Frame":
        cap = cv2.VideoCapture(video_path)
        frame_idx = st.slider("Frame", 0,
                              max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0),
                              0, key=f"frame_{key}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            annotated, dets = detect_frame(frame)
            plates = run_ocr_on_frame(frame)
            if plates:
                for p in plates:
                    cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                    cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ph.image(annotated, channels="BGR", use_container_width=True); st.session_state.last_annotated = annotated; st.session_state.last_dets = dets
            persons = sum(1 for d in dets if d["class"] == "person")
            vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
            st.caption(f"🔍 Frame {frame_idx} | Persons: {persons} | Vehicles: {vehicles} | Plates: {len(plates)}")
        cap.release()
    else:
        stride = st.slider("Process every Nth frame", 1, 10, 2, key=f"stride_{key}",
                           help="Higher = faster on CPU. Alerts still fire live.")
        col_p1, col_p2 = st.columns(2)
        if col_p1.button("▶ Play", use_container_width=True, key=f"play_{key}"):
            st.session_state[f"playing_{key}"] = True
        if col_p2.button("⏹ Stop", use_container_width=True, key=f"stop_{key}"):
            st.session_state[f"playing_{key}"] = False
            st.info("⏸ Stopped. Press Play to resume from the same position.")
        if st.session_state.get(f"playing_{key}"):
            if total <= 0:
                st.warning("⚠️ Could not read this video's frame count.")
                st.session_state[f"playing_{key}"] = False
            else:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.get(f"play_pos_{key}", 0))
                prog = st.progress(0)
                status = st.empty()
                idx = st.session_state.get(f"play_pos_{key}", 0)
                dets = []
                plates = []
                while idx < total:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    idx += 1
                    if (idx % stride) != 0:
                        continue
                    st.session_state.frame_count = idx
                    annotated, dets = detect_frame(frame)
                    plates = run_ocr_on_frame(frame)
                    if plates:
                        for p in plates:
                            cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                            cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    ph.image(annotated, channels="BGR", use_container_width=True); st.session_state.last_annotated = annotated; st.session_state.last_dets = dets
                    persons = sum(1 for d in dets if d["class"] == "person")
                    vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                    status.caption(f"🔍 Frame {idx}/{total} | Persons: {persons} | Vehicles: {vehicles} | Plates: {len(plates)}")
                    prog.progress(min(idx / total, 1.0))
                    st.session_state[f"play_pos_{key}"] = idx
                cap.release()
                st.session_state[f"play_pos_{key}"] = 0
                st.session_state[f"playing_{key}"] = False
                st.success("✅ Video finished — alerts are in the Alert Log.")
        else:
            st.info("▶ Press Play to run live detection on this video.")


# ═══════════════════════════════════════════════════
# UI HELPERS — tactical command-center components
# ═══════════════════════════════════════════════════

def _chip(label, kind="info"):
    return f"<span class='chip chip-{kind}'>● {label}</span>"


def _kpi(label, value, state="ok", state_text=None):
    stc = {"ok": "ok", "warn": "warn", "crit": "crit"}.get(state, "ok")
    stxt = state_text or {"ok": "● ONLINE", "warn": "▲ WATCH", "crit": "❗ ATTENTION"}[stc]
    return (f"<div class='kpi'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-val'>{value}</div>"
            f"<div class='kpi-state {stc}'>{stxt}</div></div>")


def _kpi_row(items):
    """Render a row of tactical KPI cards from (label, value, state, state_text)."""
    cols = st.columns(len(items))
    for col, (label, value, state, stext) in zip(cols, items):
        with col:
            st.markdown(_kpi(label, value, state, stext), unsafe_allow_html=True)


def _feed_header_html(cam, loc, tags):
    tags_html = "".join(f"<span class='fh-tag'>{t}</span>" for t in tags)
    return (f"<div class='feed-head'><div><span class='fh-id'>{cam}</span> "
            f"<span class='fh-loc'> · {loc}</span></div><div>{tags_html}</div></div>")


def _alert_card_html(a, acked=False):
    sev = a.get("severity", "low")
    cls = {"critical": "crit", "high": "high", "medium": "med",
           "low": "low"}.get(sev, "low")
    icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(sev, "⚪")
    risk = a.get("risk_score")
    risk_html = ""
    if risk is not None:
        rcol = "#DC2626" if risk >= 80 else ("#D97706" if risk >= 60 else "#16A34A")
        risk_html = (f"<span class='ac-risk' style='color:{rcol}'>"
                     f"THREAT {risk}/100</span>")
    warn = "⚠️ " if a.get("simulated") else ""
    ack_html = ("<span class='ack-chip'>✔ ACKED</span>" if acked else
                "<span class='ack-chip ack-chip-new'>NEW</span>")
    return (f"<div class='acard {cls}'><div class='ac-title'>{icon} {warn}"
            f"{a.get('event_type', '').upper()}</div>{ack_html}{risk_html}"
            f"<div class='ac-meta'>{a.get('camera_id', 'CAM-01')} · "
            f"{a.get('site_id', 'BOP-01')} · "
            f"{str(a.get('timestamp', ''))[:19]}</div>"
            f"<div class='ac-expl'>{a.get('explanation', '')}</div></div>")


def _sys_stat_bar():
    """Compact system status strip shown above every section."""
    on = sum(1 for v in st.session_state.get("camera_status", {}).values() if v)
    total = max(1, len(st.session_state.get("camera_status", {})))
    link_ok = st.session_state.get("link_up", True)
    ll = st.session_state.get("link_up", True)
    net_cls = "ok" if link_ok else "warn"
    net_txt = "NETWORK CONNECTED" if link_ok else "EDGE MODE — OFFLINE QUEUE"
    sys_cls = "ok" if on else "warn"
    cams = f"<b>{on}/{total}</b>"
    alr = len(st.session_state.get("alerts", []))
    crt = sum(1 for a in st.session_state.get("alerts", [])
              if a.get("severity") == "critical")
    return (
        "<div class='cc-head'><div><div class='cc-title'>BORDEREYE</div>"
        "<div class='cc-sub'>Border Intelligence &amp; Video Analytics Engine — "
        "AI-powered surveillance · Existing CCTV · Edge Analytics</div></div>"
        "<div class='cc-stats'>"
        f"<span class='cc-stat'><span class='dot dot-{sys_cls}'></span>SYSTEM "
        f"{'ONLINE' if sys_cls == 'ok' else 'DEGRADED'}</span>"
        f"<span class='cc-stat'><span class='dot dot-{net_cls}'></span>{net_txt}</span>"
        f"<span class='cc-stat'>CAMERAS {cams}</span>"
        f"<span class='cc-stat'>ALERTS <b>{alr}</b></span>"
        f"<span class='cc-stat'>CRIT <b style='color:{'#DC2626' if crt else '#16A34A'}'>"
        f"{crt}</b></span>"
        "</div></div>")


def _beep_datauri(freq=880.0, dur=0.18, vol=0.35):
    """16-bit PCM sine beep → data:audio/wav;base64 (stdlib only)."""
    import base64
    import io
    import math
    import struct
    import wave as _wave
    sr, n = 22050, int(22050 * dur)
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            env = max(0.0, min(1.0, i / (sr * 0.01), (n - i) / (sr * 0.02)))
            v = int(32767 * vol * env * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _beep(freq=880.0):
    """Play a short beep (best-effort; browsers gate sound on first gesture)."""
    try:
        import streamlit.components.v1 as components
        components.html(
            f"<audio autoplay='true' src='{_beep_datauri(freq)}'></audio>",
            height=0)
    except Exception:
        pass


def _ticker_html():
    """Scrolling strip of the most recent events."""
    alerts = st.session_state.get("alerts", [])
    if not alerts:
        return ("<div class='ticker-wrap'><span class='ticker'>"
                "<span class='tk-tag'>SYSTEM MONITORING — NO THREAT EVENTS</span>"
                "</span></div>")
    sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}
    body = "".join(
        f"<span><span class='tk-{a.get('severity', 'low')}'>"
        f"{sev_icon.get(a.get('severity', 'low'), '⚪')} "
        f"{a.get('event_type', '').upper()}</span>&nbsp;"
        f"<span class='tk-tag'>CAM {a.get('camera_id', '?')} · "
        f"{str(a.get('timestamp', ''))[11:19]}h</span></span>"
        for a in reversed(alerts[-8:]))
    # duplicate the body so the loop looks seamless
    return (f"<div class='ticker-wrap'><span class='ticker'>"
            f"{body}{body}</span></div>")


def _key_hook():
    """Global keyboard shortcuts 1..8 → jump straight to a section.

    Streamlit component iframes are same-origin, so the hook can attach a
    keydown listener on the parent document. Guarded against duplicates and
    ignores keystrokes typed into inputs/textareas.
    """
    try:
        import streamlit.components.v1 as components
        nav = ["🏠 Command Center", "📹 Camera Wall", "🧑 Face Intelligence",
               "🗺 Border Map", "🔎 ANPR", "🔊 Audio Threat",
               "📈 Analytics", "✏️ Tripwire Designer"]
        jmap = ";".join(
            f"if(k==='{i}'){{c('{name}');}}" for i, name in enumerate(nav, 1))
        components.html(
            f"""<script>
(function(){{
  var w = window.parent ? window.parent : window;
  if (w.__beHook) return; w.__beHook = 1;
  function c(name) {{
    var labels = w.document.querySelectorAll('label');
    for (var i = 0; i < labels.length; i++) {{
      var l = labels[i];
      if (!l.textContent) continue;
      if (l.textContent.indexOf(name) !== -1) {{
        var inp = l.querySelector('input');
        if (inp) {{ inp.click(); return; }}
        l.click(); return;
      }}
    }}
  }}
  w.document.addEventListener('keydown', function(e){{
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    var el = w.document.activeElement;
    if (el) {{
      var t = (el.tagName || '').toUpperCase();
      if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' ||
          el.isContentEditable) return;
    }}
    var k = e.key;
    if (k < '1' || k > '8') return;
    {jmap}
  }}, true);
}})();
</script>""",
            height=0)
    except Exception:
        pass


def _save_sites(coords: dict) -> bool:
    """Persist real camera coordinates to data/sites.json."""
    try:
        p = Path("data/sites.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(coords, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _dark_css():
    """Full dark-mode override (rendered only when the sidebar toggle is on)."""
    return """
<style>
html, body, .stApp { background: #0F172A !important; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background: #0F172A !important;
}
[data-testid="stMainBlockContainer"] { color: #E5E7EB !important; }
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4,
[data-testid="stMainBlockContainer"] h5,
[data-testid="stMainBlockContainer"] h6,
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] label,
[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stMetricLabel"],
[data-testid="stMainBlockContainer"] [data-testid="stMetricValue"],
[data-testid="stMainBlockContainer"] code,
[data-testid="stMainBlockContainer"] pre {
    color: #E5E7EB !important;
}
[data-testid="stMainBlockContainer"] input,
[data-testid="stMainBlockContainer"] textarea {
    color: #F3F4F6 !important;
    background: #1F2937 !important;
}
[data-testid="stMainBlockContainer"] [data-baseweb="input"],
[data-testid="stMainBlockContainer"] [data-baseweb="select"] > div {
    background: #1F2937 !important;
    border-color: #374151 !important;
    color: #F3F4F6 !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stFileUploader"] section {
    background: #1F2937 !important;
    border-color: #374151 !important;
}
.kpi, .acard, .panel, .cam-card, .cc-head, .hcell,
[data-testid="stMetric"],
[data-testid="stExpander"] details {
    background: #1F2937 !important;
    border-color: #374151 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
}
.kpi .kpi-val, .cam-id, .acard .ac-title, .cc-head .cc-title,
.hcell .hc-val { color: #F9FAFB !important; }
.kpi .kpi-label, .acard .ac-meta, .cc-head .cc-sub, .cam-loc,
.tech, .hcell .hc-tag { color: #D1D5DB !important; }
.ticker-wrap { background: #0B1220 !important; }
[data-testid="stSidebar"] { background: #0B1120 !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: #1F2937 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: #1F2937 !important; }
.stButton > button {
    background: #1F2937 !important;
    color: #F9FAFB !important;
    border: 1px solid #374151 !important;
}
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    border-color: #2563EB !important;
}
[data-testid="stTable"], [data-testid="stDataFrame"] { color: #E5E7EB !important; }
hr { border-color: #374151 !important; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #374151; border-color: #0F172A; }
</style>
"""


# ═══════════════════════════════════════════════════
# AUTH GATE — full app lock behind login
# ═══════════════════════════════════════════════════

LOGIN_QUOTES = [
    ("The border never sleeps \u2014 and neither do we.", "Field Doctrine 01"),
    ("Every alert in time is a crisis avoided.", "Field Doctrine 02"),
    ("Vigilance is the price of every peaceful dawn.", "Field Doctrine 03"),
    ("From the remotest post to command \u2014 every second, every frame counts.",
     "Field Doctrine 04"),
    ("Technology watches where eyes cannot reach.", "Field Doctrine 05"),
    ("\u0938\u0924\u0930\u094d\u0915 \u0938\u0940\u092e\u093e, \u0938\u0941\u0930\u0915\u094d\u0937\u093f\u0924 \u0926\u0947\u0936\u0964",
     "Field Doctrine 06"),
]
QUOTE_SLOT_S = 6  # seconds per quote


def login_quotes_html():
    """Rotating login quotes — pure CSS fade cycle, no reruns, no server load."""
    total = QUOTE_SLOT_S * len(LOGIN_QUOTES)
    divs = []
    for i, (text, attr) in enumerate(LOGIN_QUOTES):
        divs.append(
            f"<div class='lq lq{i}'><div class='lq-text'>\u201c{text}\u201d</div>"
            f"<div class='lq-attr'>\u2014 {attr}</div></div>"
        )
    delays = "\n".join(
        f".lq{i} {{ animation-delay: {i * QUOTE_SLOT_S}s; }}" for i in range(len(LOGIN_QUOTES))
    )
    return f"""
<style>
.lq-wrap {{ position: relative; min-height: 96px; margin: 4px 0 8px 0; }}
.lq {{ position: absolute; inset: 0; opacity: 0; text-align: center;
       animation: lqfade {total}s ease-in-out infinite; }}
.lq-text {{ font-style: italic; font-size: 0.95rem; color: #17202A; line-height: 1.45; }}
.lq-attr {{ margin-top: 6px; font-size: 0.7rem; letter-spacing: 0.18em;
            color: #0369A1; font-weight: 600; }}
{delays}
@keyframes lqfade {{
  0% {{ opacity: 0; transform: translateY(10px); }}
  2.5% {{ opacity: 1; transform: translateY(0); }}
  14% {{ opacity: 1; transform: translateY(0); }}
  16.6% {{ opacity: 0; transform: translateY(-8px); }}
  100% {{ opacity: 0; }}
}}
</style>
<div class="lq-wrap">{''.join(divs)}</div>
"""


if not st.session_state.get("authed"):
    # Header block (logo, title, strapline) sits in the 1-2-1 middle column but
    # Streamlit renders every block left-aligned — centre those first three
    # children so the lockup lines up on the page axis with the doctrine slogan
    # rotating below it. Scoped to the login card only (gone once authed).
    st.markdown(
        """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stColumn"]:has(h2) > div[data-testid="stVerticalBlock"]
    > div[data-testid="stElementContainer"]:nth-child(-n+3) { text-align: center; }
div[data-testid="stColumn"]:has(h2) [data-testid="stFullScreenFrame"] > div
    { margin-left: auto; margin-right: auto; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown("")
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        st.markdown("## 🛡️ BorderEye")
        st.caption("Border Intelligence & Video Analytics Engine")
        st.markdown(login_quotes_html(), unsafe_allow_html=True)
        st.divider()
        ADMIN_EMAIL = "admin@gmail.com"
        ADMIN_PASS = "123456"
        login_email = st.text_input("Email", key="login_email",
                                    placeholder=ADMIN_EMAIL)
        login_pass = st.text_input("Password", type="password", key="login_pass",
                                   placeholder="Enter 6-digit password", max_chars=6)
        if st.button("🔐 Login", use_container_width=True, key="login_btn"):
            if not login_email or not login_email.strip():
                st.error("Email is required.")
            elif (login_email.strip().lower() != ADMIN_EMAIL
                  or login_pass != ADMIN_PASS):
                st.error("Invalid credentials. Only the border admin account can log in.")
            else:
                st.session_state.authed = True
                st.session_state.login_user = login_email.strip().lower()
                st.rerun()
        st.divider()
        st.caption("🔒 Restricted access — border admin only")
    st.stop()

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=130)
    st.markdown("## 🛡️ BorderEye")
    st.caption("Border Intelligence & Video Analytics Engine")
    _user = st.session_state.get("login_user", "operator")
    st.caption(f"👤 Logged in as: **{_user}**")
    st.toggle("🌙 Dark Mode",
              value=st.session_state.get("dark_mode", False),
              key="dark_mode",
              help="Toggles the full dark tactical theme.")
    if st.session_state.get("dark_mode"):
        st.markdown(_dark_css(), unsafe_allow_html=True)
    st.toggle("🔊 Alert Sounds",
              value=st.session_state.get("sound_alerts", True),
              key="sound_alerts",
              help="Play a beep when a NEW CRITICAL alert lands "
                   "(browsers need one click before sound can start).")
    if st.button("▶ Test Beep", use_container_width=True, key="beep_test"):
        _beep()
        st.caption("Beep sent — if silent, click anywhere in the app once, "
                   "then test again (browser autoplay policy).")
    st.divider()

    # ── Navigation ──
    st.caption("**NAVIGATION**")
    st.radio(
        "Go to",
        ["🏠 Command Center", "📹 Camera Wall", "🧑 Face Intelligence",
         "🗺 Border Map", "🔎 ANPR", "🔊 Audio Threat",
         "📈 Analytics", "✏️ Tripwire Designer"],
        index=["🏠 Command Center", "📹 Camera Wall", "🧑 Face Intelligence",
               "🗺 Border Map", "🔎 ANPR", "🔊 Audio Threat",
               "📈 Analytics", "✏️ Tripwire Designer"].index(
                   st.session_state.get("nav_section", "🏠 Command Center")),
        key="nav_section",
        label_visibility="collapsed",
    )
    st.divider()

    # Model status
    if USE_YOLO and FINE_TUNED:
        st.success("✅ YOLOv8 fine-tuned on IDD")
        st.caption(f"7 classes incl. autorickshaw · conf {DETECT_CONF}")
    elif USE_YOLO:
        st.success("✅ YOLOv8 loaded (stock COCO)")
    elif ENGINE == "HOG":
        st.warning("⚠️ Using HOG fallback (no torch)")
    else:
        st.warning("⚠️ Using motion fallback (no torch, OpenCV 5+)")
    st.info("ℹ️ ANPR/OCR loads on first plate scan")
    st.checkbox("🌙 AI Low-Light / Night Vision", key="night_mode",
                help="Software enhancement (CLAHE) — brightens dark frames before detection. "
                     "Applies to every source (simulated/upload/webcam/RTSP). Auto-active 19:00–06:00.")
    if is_night_hours() and not st.session_state.get("night_mode"):
        st.caption("🌙 Auto-night active (time-based) — movement will raise alerts.")
    st.caption("Software night vision only — not IR/thermal hardware.")
    st.checkbox("🧑 Face Detection", key="face_detect",
                help="Draws a blue FACE box on detected faces. Built-in OpenCV — basic accuracy.")
    if st.session_state.get("face_detect") and face_clf is None:
        st.warning("Face model files missing — face detection off hai.")

    st.divider()
    if st.button("🚨 Simulate Fence Intrusion", use_container_width=True):
        create_alert("fence_intrusion",
                     "SIMULATED: Person crossed Zone-1 at 1.4 m/s, bearing NE. No patrol scheduled.",
                     "high", confidence=0.88, object_class="person", in_fence=True)
    if st.button("🚗 Simulate ANPR Match", use_container_width=True):
        import random
        plate = f"BR{random.randint(10,99)}AB{random.randint(1000,9999)}"
        create_alert("anpr_match",
                     f"SIMULATED: Vehicle {plate} flagged — plate matches watchlist at Checkpoint-1.",
                     "medium",
                     {"plate_text": plate},
                     confidence=0.85, object_class="vehicle")
    if st.button("📡 Simulate Signal Loss", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = False
        create_alert("signal_loss",
                     "SIMULATED: CAM-03 at BOP-01 lost signal. Possible jamming or tampering.",
                     "critical")
    if st.button("🔄 Restore Camera", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = True
        create_alert("signal_restored", "CAM-03 signal restored.", "low")
    if st.button("🗑️ Clear All Alerts", use_container_width=True):
        st.session_state.alerts = []
        st.session_state.prev_hash = "0" * 64

    st.divider()
    st.subheader("📡 Low-Bandwidth Link")
    st.caption("OFF = edge keeps working, events queue locally.")
    st.toggle("Link Up", value=True, key="link_up")
    # Auto-sync the moment the link restores
    if st.session_state.link_up and not st.session_state.get("link_prev", True):
        n = outbox_flush()
        if n:
            st.success(f"✅ Link restored — {n} queued event(s) synced.")
    st.session_state.link_prev = bool(st.session_state.link_up)
    q = sum(1 for i in st.session_state.get("outbox", []) if i["status"] == "queued")
    s = sum(1 for i in st.session_state.get("outbox", []) if i["status"] == "synced")
    st.caption(f"📤 Queued: {q} | ✅ Synced: {s}")
    if q and st.session_state.link_up:
        if st.button("🔄 Sync Now", use_container_width=True):
            st.success(f"✅ {outbox_flush()} event(s) synced.")
            st.rerun()

    st.divider()
    st.subheader("🆘 SOS — Higher Command")
    st.text_input("HQ webhook URL (optional)", key="sos_webhook",
                  placeholder="https://hq.example.com/api/sos")
    if st.button("🆘 SEND SOS", use_container_width=True, type="primary"):
        a = create_alert(
            "sos",
            "SOS raised by on-duty operator at BOP-01. Immediate attention "
            "from higher command requested.",
            "critical",
            payload={"raised_by": st.session_state.get("login_user", "operator")},
            confidence=0.99,
        )
        st.session_state.sos_active = True
        st.session_state.sos_time = a["timestamp"]
        url = (st.session_state.get("sos_webhook") or "").strip()
        if url:
            try:
                import requests
                r = requests.post(url, json={
                    "event_id": a["event_id"], "type": "sos",
                    "site": "BOP-01", "timestamp": a["timestamp"],
                    "raised_by": st.session_state.get("login_user", "operator"),
                }, timeout=5)
                st.session_state.sos_hook = f"HTTP {r.status_code}"
            except Exception as e:
                st.session_state.sos_hook = f"failed ({e.__class__.__name__})"
                st.warning("Webhook unreachable — SOS queued in local outbox.")
        st.rerun()
    if st.session_state.get("sos_active"):
        st.error(f"🆘 SOS ACTIVE since {st.session_state.get('sos_time', '')[:19]}")
        if st.session_state.get("sos_hook"):
            st.caption(f"Webhook: {st.session_state.sos_hook}")
        if st.button("Stand Down", use_container_width=True):
            st.session_state.sos_active = False
            create_alert("signal_restored", "SOS stood down by operator.", "low")
            st.rerun()

    st.divider()
    st.subheader("🎞 Footage")
    st.caption("Add videos (max 4) — play them in the Upload Video tab.")
    uploads = st.file_uploader("Add videos", type=["mp4", "avi", "mov", "mkv"],
                               accept_multiple_files=True, key="footage_up",
                               label_visibility="collapsed")
    if uploads:
        lib = st.session_state.get("video_lib", {})
        for up in uploads[:4]:
            k = f"{up.name}_{up.size}"
            if k not in lib and len(lib) < 4:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(up.getvalue())
                lib[k] = {"name": up.name, "path": tfile.name, "location": ""}
        st.session_state.video_lib = lib
    for k in list(st.session_state.get("video_lib", {}).keys()):
        c1, c2 = st.columns([4, 1])
        nm = st.session_state.video_lib[k].get("name", "Video")[:28]
        lc = st.session_state.video_lib[k].get("location", "")
        c1.caption(f"🎬 {nm}" + (f" • 📍 {lc[:20]}" if lc else ""))
        if c2.button("❌", key=f"delvid_{k}"):
            try:
                Path(st.session_state.video_lib[k]["path"]).unlink(missing_ok=True)
            except Exception:
                pass
            del st.session_state.video_lib[k]
            st.rerun()

    st.divider()
    st.subheader("🌐 RTSP Streams")
    st.caption("Save streams (max 4) — press Connect to go live.")
    if "rtsp_lib" not in st.session_state:
        st.session_state.rtsp_lib = [
            "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"]
    new_rtsp = st.text_input("Add RTSP URL", key="rtsp_new",
                             placeholder="rtsp://user:pass@ip:port/stream")
    if st.button("➕ Save Stream", use_container_width=True):
        url = (new_rtsp or "").strip()
        if not url.lower().startswith("rtsp://"):
            st.warning("URL `rtsp://` se start hona chahiye.")
        elif url not in st.session_state.rtsp_lib and len(st.session_state.rtsp_lib) < 4:
            st.session_state.rtsp_lib.append(url)
            meta = st.session_state.get("rtsp_meta", {})
            meta.setdefault(url, {"name": f"CAM-{len(st.session_state.rtsp_lib):02d}",
                                  "location": ""})
            st.session_state.rtsp_meta = meta
            st.rerun()
    for i, url in enumerate(list(st.session_state.get("rtsp_lib", []))):
        meta = st.session_state.get("rtsp_meta", {}).get(url, {})
        nm = meta.get("name", "")
        lc = meta.get("location", "")
        label = f"📡 {nm}" if nm else "📡 Stream"
        if lc:
            label += f" • 📍 {lc[:20]}"
        st.caption(label)
        st.caption(f"{url[:44]}")
        c1, c2 = st.columns(2)
        if c1.button("🔌 Connect", use_container_width=True, key=f"rtsp_go_{i}"):
            st.session_state.rtsp_url = url
            st.session_state.source_mode = "🌐 RTSP Stream"
            st.rerun()
        if c2.button("❌", use_container_width=True, key=f"rtsp_del_{i}"):
            st.session_state.rtsp_lib.remove(url)
            st.rerun()

    st.divider()
    st.subheader("🔐 Edge Ledger")
    st.caption("Hash-chained, tamper-evident event ledger.")
    chain_ok = verify_chain()
    if chain_ok:
        st.success("✅ Chain VALID")
    else:
        st.error("❌ Chain BROKEN!")
    st.metric("Total Events", len(st.session_state.alerts))
    if st.session_state.prev_hash != "0" * 64:
        st.caption(f"Head: `{st.session_state.prev_hash[:20]}…`")

    st.divider()
    st.subheader("⚓ Blockchain Anchor")
    st.caption("Checkpoint the day's head-hash. With a timestamp URL "
               "(e.g. Polygon Amoy relayer) this becomes public proof.")
    st.text_input("Timestamp service URL (optional)", key="anchor_url",
                  placeholder="https://relayer.example.com/anchor")
    if st.button("⚓ Anchor Now", use_container_width=True):
        _a = anchor_now()
        st.success(f"Anchored `{_a['anchor_hash'][:16]}…` ({_a['receipt']})")
    _anchors = st.session_state.get("anchors", [])
    if _anchors:
        _last = _anchors[-1]
        st.caption(f"Anchors: {len(_anchors)} | Latest: `{_last['anchor_hash'][:16]}…`")
        if st.button("Verify Anchors", use_container_width=True):
            if verify_anchors():
                st.success(f"✅ {len(_anchors)} anchor(s) intact")
            else:
                st.error("❌ Anchor log compromised!")

    st.divider()
    st.subheader("📄 Reports")
    if st.button("📄 Get Instant Alert Report (PDF)", use_container_width=True):
        try:
            st.session_state.report_pdf = build_alert_pdf()
            st.session_state.report_name = (
                "BorderEye_alerts_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            st.success("Report ready — download below.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    if st.session_state.get("report_pdf"):
        st.download_button(
            "⬇ Download PDF", data=st.session_state.report_pdf,
            file_name=st.session_state.report_name,
            mime="application/pdf", use_container_width=True,
            key="dl_report")
        with st.expander("👁 Report preview (metadata cardiac summary)"):
            _al = st.session_state.get("alerts", [])
            st.caption(f"Covers {len(_al)} alert event(s).")
            for _s in ("critical", "high", "medium", "low"):
                _n = sum(1 for a in _al if a.get("severity") == _s)
                st.markdown(f"- **{_s.title()}**: {_n}")
            st.divider()
            st.caption("Latest events in the report:")
            for a in reversed(_al[-5:]):
                st.markdown(f"{'🔴' if a.get('severity') == 'critical' else '•'} "
                            f"<span class='tech'>{a.get('event_type', '')}</span> "
                            f"· {a.get('camera_id', 'CAM-01')} · "
                            f"{str(a.get('timestamp', ''))[:19]}",
                            unsafe_allow_html=True)

    st.divider()
    st.subheader("🤖 AI Summarizer")
    st.caption("Groq LLM over alert metadata. HQ mode only — "
               "metadata leaves the device for this call.")
    if "groq_key" not in st.session_state:
        try:
            st.session_state.groq_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            st.session_state.groq_key = ""
    st.text_input("Groq API key", type="password", key="groq_key")
    _n = st.number_input("Summarize last N alerts", 1, 5, 1, key="ai_n")
    if st.button("✨ Summarize", use_container_width=True):
        if not (st.session_state.get("groq_key") or "").strip():
            st.warning("API key missing.")
        elif not st.session_state.alerts:
            st.warning("No alerts to summarize yet.")
        else:
            with st.spinner("Asking Groq…"):
                try:
                    st.session_state.ai_summary = groq_summarize(
                        st.session_state.alerts[-int(_n):],
                        st.session_state.groq_key.strip())
                except Exception as e:
                    st.session_state.ai_summary = (
                        f"AI call failed ({e.__class__.__name__}): {e}")
            st.rerun()

    st.divider()
    st.subheader("🗺 Map Coordinates")
    st.caption("Real camera lat/lng → the Border Map plots them for real. "
               "Saved to data/sites.json.")
    _mc_default = json.dumps(st.session_state.get("map_coords", {}), indent=2)
    st.text_area("sites.json (JSON)", value=_mc_default, height=120,
                 key="map_coords_json",
                 label_visibility="collapsed",
                 help="Format: {\"CAM-01\": {\"lat\": 28.62, \"lng\": 77.08, "
                      "\"name\": \"North Gate\"}}")
    if st.button("💾 Apply & Save sites.json", use_container_width=True,
                 key="map_coords_save"):
        try:
            _parsed = json.loads(st.session_state.get("map_coords_json") or "{}")
            if not isinstance(_parsed, dict):
                raise ValueError("top level must be an object")
            _clean = {k: v for k, v in _parsed.items() if isinstance(v, dict)}
            st.session_state.map_coords = _clean
            _ok = _save_sites(_clean)
            st.success("Map coordinates updated" +
                       (" & saved to data/sites.json." if _ok else
                        " (file save failed — working from session only)."))
            st.rerun()
        except Exception as _e:
            st.error(f"Invalid JSON: {_e}")

    st.divider()
    st.subheader("📡 Cameras")
    for cam, ok in st.session_state.camera_status.items():
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} {cam}")

    # ── System status footer + Logout ──
    st.divider()
    _sys_engine = "AI ENGINE ONLINE" if (USE_YOLO and model is not None) or ENGINE in ("HOG", "YOLOv8 (IDD fine-tuned)", "YOLOv8 (stock COCO)") else "DETECTOR DEGRADED"
    _sys_cls = "ok" if "ONLINE" in _sys_engine else "warn"
    _net_ok2 = st.session_state.get("link_up", True)
    _net_txt2 = "NETWORK CONNECTED" if _net_ok2 else "EDGE MODE — OFFLINE QUEUE"
    _net_cls2 = "ok" if _net_ok2 else "warn"
    st.markdown(
        f"<div class='sys-foot'><div class='sf-title'>SYSTEM STATUS</div>"
        f"<div class='sf-row'><span class='dot dot-{_sys_cls}'></span>{_sys_engine}</div>"
        f"<div class='sf-row'><span class='dot dot-{_net_cls2}'></span>{_net_txt2}</div>"
        "</div>", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ═══════════════════════════════════════════════════
# ══════════════════════════════
# SECTION RENDERERS (wrapped from the single-page layout)
# ══════════════════════════════

def render_banners():
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(_sys_stat_bar(), unsafe_allow_html=True)
    st.markdown(_ticker_html(), unsafe_allow_html=True)
    st.caption("All video analytics run on-device. Only compact alert metadata "
               "leaves the edge post.")
    _key_hook()
    st.caption("⌨️ Shortcuts: 1 Command Center · 2 Camera Wall · 3 Face · 4 Map · "
               "5 ANPR · 6 Audio · 7 Analytics · 8 Tripwire")
    if not st.session_state.get("link_up", True):
        st.warning("🔴 LINK DOWN — edge mode: AI running locally, events queuing in local outbox. Video stays on device.")
    if st.session_state.get("sos_active"):
        st.error(f"🆘 SOS ACTIVE — higher command alerted at {st.session_state.get('sos_time', '')[:19]}. Stand down from the sidebar.")



def render_command_center():
    hc1, hc2 = st.columns([5, 1])
    with hc1:
        _t1, _t2 = st.columns([1, 8])
        with _t1:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=76)
        with _t2:
            st.markdown("## 🛡️ BORDEREYE")
            st.caption("Border Intelligence & Video Analytics Engine")
            st.caption("AI-powered surveillance • Existing CCTV • Edge Analytics")
    with hc2:
        st.write("")
        st.markdown(
            "<div style='text-align:right;'>"
            "<span class='chip chip-live'>● SYSTEM ONLINE</span><br>"
            "<span style='font-size:0.68rem; color:#667085;'>SITE BOP-01</span>"
            "</div>", unsafe_allow_html=True)
        st.write("")
        with st.popover("➕ Add Camera / Footage", use_container_width=True):
            stype = st.selectbox("Source type", ["🌐 RTSP Stream", "🎞 Video Footage"],
                                 key="add_type")
            if stype == "🌐 RTSP Stream":
                st.text_input("Camera name", key="add_cam_name", placeholder="e.g. CAM-04")
                st.text_input("Location", key="add_cam_loc", placeholder="e.g. Gate-East")
                st.text_input("RTSP URL", key="add_cam_url", placeholder="rtsp://user:pass@ip:port/stream")
                if st.button("Save Camera", use_container_width=True, key="add_cam_go"):
                    url = (st.session_state.get("add_cam_url") or "").strip()
                    name = (st.session_state.get("add_cam_name") or "").strip() or "CAM"
                    loc = (st.session_state.get("add_cam_loc") or "").strip()
                    if not url.lower().startswith("rtsp://"):
                        st.error("URL must start with rtsp://")
                    elif url not in st.session_state.get("rtsp_lib", []) and len(st.session_state.get("rtsp_lib", [])) < 4:
                        st.session_state.rtsp_lib.append(url)
                        meta = st.session_state.get("rtsp_meta", {})
                        meta[url] = {"name": name, "location": loc}
                        st.session_state.rtsp_meta = meta
                        st.success(f"Camera '{name}' saved.")
                        st.rerun()
                    else:
                        st.warning("Already saved or limit (4) reached.")
            else:
                up = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"],
                                      key="add_vid")
                st.text_input("Name", key="add_vid_name", placeholder="e.g. Gate footage")
                st.text_input("Location", key="add_vid_loc", placeholder="e.g. Gate-East")
                if st.button("Save Footage", use_container_width=True, key="add_vid_go"):
                    if up is None:
                        st.error("Please choose a video file first.")
                    else:
                        lib = st.session_state.get("video_lib", {})
                        k = f"{up.name}_{up.size}"
                        name = (st.session_state.get("add_vid_name") or "").strip() or up.name
                        loc = (st.session_state.get("add_vid_loc") or "").strip()
                        if k not in lib and len(lib) < 4:
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                            tfile.write(up.getvalue())
                            lib[k] = {"name": name, "path": tfile.name, "location": loc}
                            st.session_state.video_lib = lib
                            st.success(f"Footage '{name}' saved.")
                            st.rerun()
                        else:
                            st.warning("Already saved or limit (4) reached.")


    # Tactical KPI row
    _cam_ok = sum(1 for v in st.session_state.camera_status.values() if v)
    _alerts_n = len(st.session_state.alerts)
    _critical_n = sum(1 for a in st.session_state.alerts if a["severity"] == "critical")
    _tot = st.session_state.get("stats_total", {})
    _veh = sum(int(_tot.get(c, 0)) for c in ("car", "bus", "truck", "motorcycle"))
    _kpi_row([
        ("ACTIVE CAMERAS", f"{_cam_ok:02d}", "ok", "● ONLINE"),
        ("ACTIVE ALERTS", f"{_alerts_n:02d}",
         "warn" if _alerts_n else "ok",
         "▲ ACTIVE" if _alerts_n else "● MONITORING"),
        ("CRITICAL EVENTS", f"{_critical_n:02d}",
         "crit" if _critical_n else "ok",
         "❗ REQUIRES ATTENTION" if _critical_n else "● CLEAR"),
        ("PERSONS DETECTED", f"{int(_tot.get('person', 0)):02d}", "ok", "● TRACKED"),
        ("VEHICLES DETECTED", f"{_veh:02d}", "ok", "● TRACKED"),
        ("SYSTEM", "READY", "ok", "● OPERATIONAL"),
    ])
    st.divider()

    # 24-hour activity sparkline (thin bar chart from the hourly counters)
    st.markdown(
        "<div class='panel'><div class='panel-title'>24-HOUR ACTIVITY "
        "TREND</div>", unsafe_allow_html=True)
    _hh = st.session_state.get("stats_hourly", {})
    _now_dt = datetime.now()
    _labels = [(_now_dt - timedelta(hours=i)).strftime("%H:00")
               for i in range(23, -1, -1)]
    _vals = [int(_hh.get(h, 0)) for h in _labels]
    _total_24 = sum(_vals)
    st.caption("Detections per hour over the last 24h — run any source to fill "
               "the curve.")
    if _total_24:
        st.bar_chart({"detections": _vals}, height=110, use_container_width=True)
        _pk = max(_vals)
        _pk_hrs = ", ".join(_labels[i] for i, v in enumerate(_vals) if v == _pk)
        st.caption(f"Peak activity: {_pk} detections/h at {_pk_hrs} · "
                   f"24h total: {_total_24}")
    else:
        st.caption("No hourly data yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # System health strip
    _up = int(time.time() - st.session_state.get("_started", time.time()))
    _up_h, _up_r = divmod(_up, 3600)
    _up_m, _up_s = divmod(_up_r, 60)
    _fr = st.session_state.get("frame_count", 0)
    _fps = (_fr / _up) if _up else 0.0
    _q = sum(1 for i in st.session_state.get("outbox", [])
             if i.get("status") == "queued")
    _head = st.session_state.get("prev_hash", "0" * 64)[:12]
    st.markdown(
        "<div class='panel'><div class='panel-title'>SYSTEM HEALTH</div>"
        "<div class='health-grid'>"
        f"<div class='hcell'><div class='hc-tag'>UPTIME</div>"
        f"<div class='hc-val'>{_up_h:02d}:{_up_m:02d}:{_up_s:02d}</div></div>"
        f"<div class='hcell'><div class='hc-tag'>FRAMES</div>"
        f"<div class='hc-val'>{_fr:,}</div></div>"
        f"<div class='hcell'><div class='hc-tag'>AVG FPS</div>"
        f"<div class='hc-val'>{_fps:.1f}</div></div>"
        f"<div class='hcell'><div class='hc-tag'>OUTBOX QUEUE</div>"
        f"<div class='hc-val'>{_q}</div></div>"
        f"<div class='hcell'><div class='hc-tag'>LEDGER HEAD</div>"
        f"<div class='hc-val'>{_head}</div></div>"
        "</div></div>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            "<div class='feed-head'><div><span class='fh-id'>CAM-01</span> "
            "<span class='fh-loc'> · BOP-01 · Main Analysis Feed</span></div>"
            "<div>"
            "<span class='fh-tag'><span class='live-dot'></span>LIVE</span> "
            "<span class='rec-tag'><span class='rec-dot'></span>REC</span> "
            "<span class='fh-tag'>AI ANALYTICS ACTIVE</span>"
            "</div></div>", unsafe_allow_html=True)

        mode = st.radio("Source", ["🎬 Simulated Feed", "🎥 Upload Video", "📷 Webcam", "🌐 RTSP Stream"],
                         horizontal=True, key="source_mode")

        placeholder = st.empty()

        if mode == "🎬 Simulated Feed":
            tick = st.session_state.frame_count
            for _ in range(3):
                frame, in_fence = generate_demo_frame(tick)
                # Run detection on synthetic frame
                annotated, dets = detect_frame(frame)
                placeholder.image(annotated, channels="BGR", use_container_width=True); st.session_state.last_annotated = annotated; st.session_state.last_dets = dets
                tick += 1
                time.sleep(0.3)
            st.session_state.frame_count = tick

            persons = sum(1 for d in dets if d["class"] == "person")
            vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
            st.caption(f"🔍 Detected: {len(dets)} objects | Persons: {persons} | Vehicles: {vehicles}")

        elif mode == "🎥 Upload Video":
            uploads_main = st.file_uploader("Upload videos (max 4)", type=["mp4", "avi", "mov", "mkv"],
                                            accept_multiple_files=True, key="footage_main")
            if uploads_main:
                lib = st.session_state.get("video_lib", {})
                for up in uploads_main[:4]:
                    k = f"{up.name}_{up.size}"
                    if k not in lib and len(lib) < 4:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(up.getvalue())
                        lib[k] = {"name": up.name, "path": tfile.name, "location": ""}
                st.session_state.video_lib = lib
            lib = st.session_state.get("video_lib", {})
            if not lib:
                st.info("📁 Upload videos above (max 4). Then play them one by one in the tabs below.")
            else:
                keys = list(lib.keys())
                tabs = st.tabs([lib[k]["name"][:20] for k in keys])
                for tab, k in zip(tabs, keys):
                    with tab:
                        render_video_tab(lib[k]["path"], k, lib[k].get("location", ""))

        elif mode == "📷 Webcam":
            # DEPLOY NOTE: Live uses the *server's* camera. Local run = your
            # laptop camera with full live tracking. On cloud deploy there is no
            # camera device, so Live shows a clear error and Photo mode (browser
            # capture via getUserMedia) keeps working. Nothing crashes.
            wview = st.radio("Webcam mode", ["🔴 Live Camera", "📸 Photo"],
                             horizontal=True, key="webcam_view")
            if wview == "📸 Photo":
                snap = st.camera_input("Take a photo", label_visibility="collapsed")
                if snap is not None:
                    file_bytes = np.frombuffer(snap.getvalue(), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if frame is not None:
                        annotated, dets = detect_frame(frame)

                        plates = run_ocr_on_frame(frame)
                        if plates:
                            for p in plates:
                                cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                                cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        placeholder.image(annotated, channels="BGR", use_container_width=True); st.session_state.last_annotated = annotated; st.session_state.last_dets = dets

                        persons = sum(1 for d in dets if d["class"] == "person")
                        vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                        st.caption(f"🔍 Detected: {len(dets)} objects | Persons: {persons} | Vehicles: {vehicles} | Plates: {len(plates)}")
                else:
                    st.info("📷 Click above to capture a photo from your camera")
            else:
                wstride = st.slider("Process every Nth frame", 1, 10, 2, key="webcam_stride",
                                    help="Higher = faster on CPU. Alerts still fire live.")
                wmax = st.number_input("Max frames per run", min_value=60, max_value=10000,
                                       value=600, step=60, key="webcam_max",
                                       help="Each run stops after this many frames. Press Start to continue.")
                wc1, wc2 = st.columns(2)
                if wc1.button("▶ Start Live", use_container_width=True, key="webcam_start"):
                    st.session_state.webcam_live = True
                if wc2.button("⏹ Stop", use_container_width=True, key="webcam_stop"):
                    st.session_state.webcam_live = False
                if st.session_state.get("webcam_live"):
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("❌ No camera found. Live mode is unavailable on cloud deployments — use 📸 Photo mode.")
                        st.session_state.webcam_live = False
                    else:
                        st.success("🟢 Live camera — YOLO + fence + ANPR + face + alerts sab live.")
                        status = st.empty()
                        n = 0
                        dets = []
                        while n < wmax:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("⚠️ Camera is not delivering frames. Press Start again.")
                                break
                            n += 1
                            if (n % wstride) != 0:
                                continue
                            st.session_state.frame_count += 1
                            annotated, dets = detect_frame(frame)

                            plates = run_ocr_on_frame(frame)
                            if plates:
                                for p in plates:
                                    cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                                    cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            placeholder.image(annotated, channels="BGR", use_container_width=True); st.session_state.last_annotated = annotated; st.session_state.last_dets = dets

                            persons = sum(1 for d in dets if d["class"] == "person")
                            vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                            faces = st.session_state.get("last_faces", 0)
                            status.caption(f"🔴 LIVE CAM | Frame {n}/{wmax} | Persons: {persons} | Vehicles: {vehicles} | Faces: {faces}")
                        cap.release()
                        st.session_state.webcam_live = False
                        st.success("✅ Run complete — alerts are in the Alert Log. Press Start to continue.")
                else:
                    st.info("▶ Press Start Live for continuous tracking — no photo capture needed.")

        elif mode == "🌐 RTSP Stream":
            if "rtsp_url" not in st.session_state:
                st.session_state.rtsp_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
            if "rtsp_live" not in st.session_state:
                st.session_state.rtsp_live = False

            saved = st.session_state.get("rtsp_lib", [])
            meta = st.session_state.get("rtsp_meta", {})
            labels = ["Custom URL"] + [
                (meta.get(u, {}).get("name") or f"Camera {j + 1}") +
                (f" ({meta.get(u, {}).get('location')})" if meta.get(u, {}).get("location") else "")
                for j, u in enumerate(saved)]
            pick = st.selectbox("Saved cameras", labels, key="rtsp_pick")
            if pick != "Custom URL":
                st.session_state.rtsp_url = saved[labels.index(pick) - 1]
            url = st.text_input("RTSP URL", value=st.session_state.rtsp_url,
                                placeholder="rtsp://user:pass@ip:port/stream")
            stride_rtsp = st.slider("Process every Nth frame", 1, 10, 2, key="rtsp_stride",
                                    help="Higher = faster on CPU. Alerts still fire live.")
            max_frames = st.number_input("Max frames per run", min_value=60, max_value=10000,
                                         value=600, step=60,
                                         help="Each run stops after this many frames. Press Connect to continue.")

            col_r1, col_r2 = st.columns(2)
            if col_r1.button("🔌 Connect + Start Live", use_container_width=True):
                st.session_state.rtsp_url = url.strip()
                st.session_state.rtsp_live = True
            if col_r2.button("⏹ Disconnect", use_container_width=True):
                st.session_state.rtsp_live = False

            if st.session_state.get("rtsp_live"):
                target = st.session_state.rtsp_url
                if not target.lower().startswith("rtsp://"):
                    st.error("❌ URL `rtsp://` se start hona chahiye.")
                    st.session_state.rtsp_live = False
                else:
                    cap = cv2.VideoCapture(target)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    except Exception:
                        pass
                    if not cap.isOpened():
                        st.error("❌ Stream could not be opened. Check the URL and connection, then retry.")
                        st.session_state.rtsp_live = False
                    else:
                        st.success("🟢 Live connected — tracking in progress.")
                        status = st.empty()
                        n = 0
                        dets = []
                        while n < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("⚠️ Stream interrupted (network/camera). Press Connect to retry.")
                                break
                            n += 1
                            if (n % stride_rtsp) != 0:
                                continue
                            st.session_state.frame_count += 1
                            annotated, dets = detect_frame(frame)
                            placeholder.image(annotated, channels="BGR", use_container_width=True); st.session_state.last_annotated = annotated; st.session_state.last_dets = dets

                            persons = sum(1 for d in dets if d["class"] == "person")
                            vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                            status.caption(f"🔴 LIVE | Frame {n}/{max_frames} | Persons: {persons} | Vehicles: {vehicles}")
                        cap.release()
                        st.session_state.rtsp_live = False
                        st.success("✅ Run complete — alerts are in the Alert Log. Press Connect to continue.")
            else:
                st.info("🔌 Paste an RTSP URL and press Connect for live tracking — no camera hardware needed.")

    with col2:
        st.markdown(
            "<div class='panel-title'>SECURITY EVENT CONSOLE</div>",
            unsafe_allow_html=True)

        # Beep on NEW critical alerts (best-effort; browsers gate sound on first click)
        _sound = st.session_state.get("sound_alerts", True)
        _seen = st.session_state.setdefault("_seen_alerts", set())
        _new_crit = [a for a in st.session_state.alerts
                     if a.get("severity") == "critical"
                     and a.get("event_id") not in _seen]
        for _a in _new_crit:
            _seen.add(_a["event_id"])
        if _new_crit and _sound:
            _beep()
            st.markdown(
                "<div class='panel'><span class='ack-chip ack-chip-new'>"
                "🔴 NEW CRITICAL ALERT</span> Operator attention required — "
                "event(s) landing in the console below. ACK to acknowledge.</div>",
                unsafe_allow_html=True)

        _acked = st.session_state.setdefault("_acked", set())
        _all_alerts = st.session_state.alerts
        _subset = []
        if _all_alerts:
            _sev_filter = st.pills(
                "Severity filter", ["Critical", "High", "Medium", "Low"],
                selection_mode="multi", key="alert_sev_filter",
                label_visibility="collapsed")
            _fmap = {"critical": "Critical", "high": "High",
                     "medium": "Medium", "low": "Low"}
            _subset = [a for a in reversed(_all_alerts[-40:])
                       if not _sev_filter
                       or _fmap.get(a.get("severity")) in _sev_filter]
            _unacked = sum(1 for a in _subset
                           if a.get("event_id") not in _acked)
            _c1, _c2 = st.columns([4, 1])
            with _c1:
                st.caption(f"{len(_subset)} shown · {_unacked} unacked")
            with _c2:
                if st.button("ACK ALL", use_container_width=True,
                             key="ack_all") and _subset:
                    _acked.update(_a2["event_id"] for _a2 in _subset)

        if not _all_alerts:
            st.markdown(
                "<div class='panel'><div class='ac-title' "
                "style='color:#667085;'>NO ACTIVE THREATS</div>"
                "<div class='ac-expl'>System monitoring continuously… "
                "No threat events on record.</div></div>",
                unsafe_allow_html=True)
        else:
            for alert in _subset:
                _ackd = alert.get("event_id") in _acked
                st.markdown(_alert_card_html(alert, acked=_ackd),
                            unsafe_allow_html=True)
                _cols = st.columns([2, 1])
                with _cols[0]:
                    _ls = alert.get("link_status", st.session_state.get(
                        "outbox_status", {}).get(alert["event_id"], "synced"))
                    if _ls == "queued":
                        st.caption("📤 Queued locally — syncs when link restores")
                    else:
                        st.caption("✅ Synced to Command Centre (metadata only)")
                with _cols[1]:
                    if not _ackd:
                        if st.button("✔ ACK", key=f"ack_{alert['event_id']}",
                                     use_container_width=True):
                            _acked.add(alert["event_id"])
                            st.rerun()
                if _ackd:
                    st.caption("✔ Acknowledged by operator — cleared from the "
                               "attention queue.")
                risk = alert.get("risk_score")
                if risk is not None:
                    with st.expander("Threat score breakdown"):
                        for r in alert.get("risk_reasons", []):
                            st.caption(f"• {r}")

        st.divider()

        # Alert object schema
        if st.session_state.alerts:
            st.subheader("📋 Latest Alert Object")
            last = st.session_state.alerts[-1].copy()
            last.pop("prev_hash", None)
            st.json(last)

        # Low-bandwidth payload: exactly what travels upstream
        if st.session_state.alerts:
            st.markdown(
                "<div class='panel'><div class='panel-title'>LOW-BANDWIDTH "
                "PAYLOAD</div>", unsafe_allow_html=True)
            st.caption("Only this metadata crosses the link — never video.")
            _a = st.session_state.alerts[-1]
            _mini = {
                "event_id": _a["event_id"], "ts": _a["timestamp"],
                "site": _a["site_id"], "cam": _a["camera_id"],
                "type": _a["event_type"], "sev": _a["severity"],
                "risk": _a.get("risk_score"),
                "track": (_a.get("payload") or {}).get("track_id"),
            }
            st.json(_mini)
            st.markdown(
                f"<span class='tech'>~{len(json.dumps(_mini))} BYTES ON WIRE — "
                "VIDEO STAYS ON EDGE</span></div>",
                unsafe_allow_html=True,
            )

        # AI summary (Groq) — rendered from sidebar action
        if st.session_state.get("ai_summary"):
            st.subheader("🤖 AI Summary")
            st.caption("Groq LLM over alert metadata — HQ mode only.")
            st.markdown(st.session_state.ai_summary)

        # Hash chain verification — tamper-evident ledger
        chain_ok = verify_chain()
        st.markdown(
            "<div class='panel'><div class='panel-title'>SHA-256 SECURITY "
            "LEDGER</div>"
            + ("<span class='chip chip-live'>● CHAIN INTEGRITY VERIFIED</span>"
               if chain_ok else
               "<span class='chip chip-crit'>● CHAIN BROKEN!</span>")
            + f" <span class='tech'>{len(st.session_state.alerts)} events "
            "hash-linked · head "
            f"{st.session_state.get('prev_hash', '0'*64)[:16]}…</span>"
            "</div>", unsafe_allow_html=True)
        if st.button("🔎 Verify Chain Integrity"):
            if verify_chain():
                st.success(f"✅ Chain valid — {len(st.session_state.alerts)} events verified")
            else:
                st.error("❌ Chain integrity compromised!")


def render_analytics():
    # ═══════════════════════════════════════════════════
    # BEHAVIORAL ANALYTICS
    # ═══════════════════════════════════════════════════

    st.divider()
    st.subheader("📈 Behavioral Analytics")
    st.caption("Live counts from every processed frame — run any source to populate the graphs.")

    tot = st.session_state.get("stats_total", {})
    frames = st.session_state.get("stats_frames", 0)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Frames Analyzed", frames)
    a2.metric("Persons Seen", tot.get("person", 0))
    a3.metric("Vehicles Seen", sum(tot.get(k, 0) for k in
             ("car", "bus", "truck", "motorcycle", "bicycle", "autorickshaw")))
    a4.metric("Faces Seen", tot.get("face", 0))

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Detections by class")
        if tot:
            st.bar_chart(tot)
        else:
            st.info("No data yet — run any source.")
    with c2:
        st.caption("Activity by hour")
        hh = st.session_state.get("stats_hourly", {})
        if hh:
            st.bar_chart(dict(sorted(hh.items())))
        else:
            st.info("No data yet.")
    if st.button("🧹 Reset Analytics"):
        st.session_state.stats_total = {}
        st.session_state.stats_hourly = {}
        st.session_state.stats_frames = 0
        st.rerun()



def render_tripwire_designer():
    # ═══════════════════════════════════════════════════
    # TRIPWIRE DESIGNER — pen-type fence (click to draw any line)
    # ═══════════════════════════════════════════════════

    st.divider()
    st.subheader("✏️ Tripwire Designer — draw your own fence")
    st.caption("Click points on the canvas to sketch any line. Save it as a tripwire — "
               "any tracked person/vehicle crossing that line fires tracking + alert.")

    _ref = np.zeros((H, W, 3), dtype=np.uint8) + 18
    for _gx in range(0, W, 40):
        cv2.line(_ref, (_gx, 0), (_gx, H), (40, 40, 40), 1)
    for _gy in range(0, H, 40):
        cv2.line(_ref, (0, _gy), (W, _gy), (40, 40, 40), 1)
    cv2.polylines(_ref, [FENCE], True, (0, 120, 0), 1)
    for _i, _seg in enumerate(st.session_state.get("tripwires", [])):
        (_x1, _y1), (_x2, _y2) = _seg
        cv2.line(_ref, (int(_x1), int(_y1)), (int(_x2), int(_y2)), (255, 0, 255), 2)
        cv2.putText(_ref, f"TW-{_i + 1}", (int(_x1) + 4, int(_y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    _pts = st.session_state.get("pen_points", [])
    for _k in range(1, len(_pts)):
        cv2.line(_ref, _pts[_k - 1], _pts[_k], (0, 255, 255), 2)
    for (_px, _py) in _pts:
        cv2.circle(_ref, (_px, _py), 4, (0, 255, 255), -1)

    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        _click = streamlit_image_coordinates(
            cv2.cvtColor(_ref, cv2.COLOR_BGR2RGB), width=W, key="trip_canvas")
        if _click is not None:
            _np = (int(_click["x"]), int(_click["y"]))
            if st.session_state.get("pen_last") != _np:
                st.session_state.pen_last = _np
                _cur = st.session_state.get("pen_points", [])
                _cur.append(_np)
                st.session_state.pen_points = _cur
                st.rerun()
    except Exception as e:
        st.warning(f"Drawing canvas unavailable ({e.__class__.__name__}). "
                   "Add tripwires with coordinates below instead.")
        _c1, _c2, _c3, _c4 = st.columns(4)
        _fx1 = _c1.number_input("x1", 0, W, 100, key="fx1")
        _fy1 = _c2.number_input("y1", 0, H, 100, key="fy1")
        _fx2 = _c3.number_input("x2", 0, W, 500, key="fx2")
        _fy2 = _c4.number_input("y2", 0, H, 400, key="fy2")
        if st.button("➕ Add line tripwire", key="fx_add"):
            st.session_state.setdefault("tripwires", []).append(
                ((int(_fx1), int(_fy1)), (int(_fx2), int(_fy2))))
            st.success("Tripwire added.")
            st.rerun()

    _dc1, _dc2, _dc3, _dc4 = st.columns(4)
    if _dc1.button("↩ Undo point", key="pen_undo"):
        _cur = st.session_state.get("pen_points", [])
        if _cur:
            _cur.pop()
            st.session_state.pen_points = _cur
            st.session_state.pen_last = None
            st.rerun()
    if _dc2.button("🗑 Clear sketch", key="pen_clear"):
        st.session_state.pen_points = []
        st.session_state.pen_last = None
        st.rerun()
    if _dc3.button("💾 Save as tripwire", key="pen_save",
                   disabled=len(st.session_state.get("pen_points", [])) < 2):
        _cur = st.session_state.get("pen_points", [])
        _tw = st.session_state.get("tripwires", [])
        for _k in range(1, len(_cur)):
            _tw.append((_cur[_k - 1], _cur[_k]))
        st.session_state.tripwires = _tw
        st.session_state.pen_points = []
        st.session_state.pen_last = None
        create_alert("signal_restored",
                     f"Tripwire-{len(_tw)} drawn by operator ({len(_cur)} points).",
                     "low", payload={"segments": len(_tw)})
        st.success(f"✅ Tripwire saved ({len(_cur) - 1} segment(s)). Crossings now tracked.")
        st.rerun()
    if _dc4.button("❌ Delete all tripwires", key="pen_delall"):
        st.session_state.tripwires = []
        st.session_state.pen_points = []
        st.session_state.pen_last = None
        st.session_state.track_hist = {}
        st.rerun()
    st.caption(f"Active tripwires: {len(st.session_state.get('tripwires', []))} "
               f"| Sketch points: {len(st.session_state.get('pen_points', []))} "
               "(canvas is 640×480 = detection frame size)")



def render_camera_wall():
    """Multi-camera security wall — one live source + labelled simulated feeds."""
    st.subheader("\U0001F4F9 Multi-Camera Security Wall")
    st.markdown(
        "<span class='chip chip-live'>● LIVE</span> "
        "<span class='chip chip-sim'>● SIMULATED FEED</span> "
        "<span class='chip chip-info'>● HONEST LABELS</span>",
        unsafe_allow_html=True)
    st.caption("Only the connected/working source is LIVE. Every other feed is "
               "clearly labelled SIMULATED — no simulated feed pretends to be a "
               "real government camera.")
    c1, c2 = st.columns(2)
    cams = [
        ("CAM-01", "North Gate \u00b7 main source"),
        ("CAM-02", "Wire Zone \u00b7 simulated"),
        ("CAM-03", "Checkpoint \u00b7 simulated"),
        ("CAM-04", "Riverine \u00b7 simulated"),
    ]
    if st.button("\U0001F504 Refresh feed previews", use_container_width=True,
                 key="wall_refresh"):
        st.session_state.setdefault("_wall_cache", {}).clear()
        st.session_state._wall_ts = time.time()
    cache = st.session_state.setdefault("_wall_cache", {})
    for idx, (cid, loc) in enumerate(cams):
        col = c1 if idx % 2 == 0 else c2
        with col:
            ok = st.session_state.get("camera_status", {}).get(cid, True)
            live_ = (cid == "CAM-01" and st.session_state.get("last_annotated") is not None)
            if live_:
                badge = ("<span class='pill pill-live'>"
                         "<span class='live-dot'></span>LIVE</span> "
                         "<span class='rec-dot'></span>"
                         "<span class='rec-tag'>REC</span>")
            elif not ok:
                badge = "<span class='pill pill-off'>OFFLINE</span>"
            else:
                badge = "<span class='chip chip-sim'>● SIMULATED FEED</span>"
            data = cache.get(cid)
            if data is None:
                if cid == "CAM-01" and st.session_state.get("last_annotated") is not None:
                    data = (st.session_state.last_annotated,
                            st.session_state.last_dets or [])
                else:
                    tick = st.session_state.get("frame_count", 0) + idx * 11
                    frame, _ = generate_demo_frame(tick)
                    annotated, dets = detect_frame(frame)
                    data = (annotated, dets)
                cache[cid] = data
                st.session_state._wall_ts = time.time()
            annotated, dets = data
            persons = sum(1 for d in dets if d["class"] == "person")
            vehicles = sum(1 for d in dets if d["class"] in VEHICLE_CLASSES)
            last_alert = "\u2014"
            for a in reversed(st.session_state.get("alerts", [])):
                if a.get("camera_id") == cid:
                    last_alert = a.get("event_type", "\u2014")
                    break
            _st_html = ("<span class='live-dot'></span>ONLINE" if ok
                        else "\U0001F534 OFFLINE")
            st.markdown(
                f"<div class='cam-card'><div class='cam-head'>"
                f"<span class='cam-id'>{cid}</span> {badge}</div>"
                f"<div class='cam-loc'>{loc} \u00b7 "
                f"{_st_html} \u00b7 "
                f"P:{persons} V:{vehicles} \u00b7 last: <b>{last_alert}</b></div></div>",
                unsafe_allow_html=True)
            st.image(annotated, channels="BGR", use_container_width=True)
            if cid == "CAM-01":
                if st.button("\u25B6 Open main live feed", key=f"wall_open_{cid}",
                             use_container_width=True):
                    st.session_state.nav_section = "\U0001F3E0 Command Center"
                    st.rerun()
            else:
                st.caption("SIMULATED feed \u2014 sample for the security-wall view.")
    _age = max(0, int(time.time() - st.session_state.get("_wall_ts", time.time())))
    st.caption(f"⏱ Previews updated {_age}s ago — honest freshness: if a feed "
               "stalls, this number keeps climbing. Press Refresh to recompute "
               "through the live detection pipeline.")


def render_anpr_section():
    """Vehicle plate intelligence — read runs + honest unclear handling."""
    st.subheader("\U0001F50E Vehicle Plate Intelligence (ANPR)")
    st.markdown(
        "<span class='chip chip-info'>● MULTI-PASS OCR</span> "
        "<span class='chip chip-warn'>● HONEST UNREADABLE HANDLING</span>",
        unsafe_allow_html=True)
    st.caption("Pipeline: vehicle detection → plate region → crop → enhance → "
               "OCR → confidence. If a plate is unreadable the system reports "
               "'Plate text unclear' — it never invents a registration number.")
    log = st.session_state.get("plate_log", [])
    if not log:
        st.info("No plates scanned yet. Run any video/webcam/RTSP source showing "
                "a vehicle plate.")
    else:
        for p in reversed(log[-25:]):
            if p.get("unclear"):
                st.markdown(
                    "⚠️ <span class='tech'>PLATE TEXT UNCLEAR</span> — "
                    "low OCR confidence", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"✅ <b>{p['text']}</b> "
                    f"<span class='tech'>conf {p['confidence']:.0%}</span>",
                    unsafe_allow_html=True)
            st.caption(f"{p['ts']} · {p.get('cam', 'CAM-01')}")
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Scans logged", len(log))
        c2.metric("Readable", sum(1 for p in log if not p.get("unclear")))
    st.caption("OCR engine loads lazily on the first plate scan (EasyOCR, "
               "English). ANPR failure never stops vehicle detection.")

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NAVIGATION DISPATCH
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

render_banners()
_SECTION = st.session_state.get("nav_section", "\U0001F3E0 Command Center")
if _SECTION == "\U0001F3E0 Command Center":
    render_command_center()
elif _SECTION == "\U0001F4F9 Camera Wall":
    render_camera_wall()
elif _SECTION == "🧑 Face Intelligence":
    if render_face_intelligence is not None:
        render_face_intelligence(create_alert)
    else:
        st.warning("Face Intelligence module unavailable (dependency missing).")
elif _SECTION == "\U0001F5FA Border Map":
    if render_border_map is not None:
        render_border_map()
    else:
        st.warning("Border Map module unavailable (folium not installed).")
elif _SECTION == "\U0001F50E ANPR":
    render_anpr_section()
elif _SECTION == "\U0001F50A Audio Threat":
    if render_audio_threat is not None:
        render_audio_threat(create_alert)
    else:
        st.warning("Audio Threat module unavailable (dependencies missing).")
elif _SECTION == "\U0001F4C8 Analytics":
    render_analytics()
elif _SECTION == "\u270F\uFE0F Tripwire Designer":
    render_tripwire_designer()

# Footer
st.divider()
st.caption("BorderEye — Smart India Hackathon 2026 | "
           "\"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust. "
           "Border posts have none of those three.\"")

