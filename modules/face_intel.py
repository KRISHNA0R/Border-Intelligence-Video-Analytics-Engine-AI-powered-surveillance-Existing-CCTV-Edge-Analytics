"""Face Intelligence — clearly-labelled DEMO face database.

Honest boundary: this module does face *detection* (Haar, in the core app)
and keeps a local demo identification database. It does NOT perform real
biometric face recognition — the UI is explicit about that, so a hackathon
demo never over-claims an identity match.

Storage: data/face_db.jsonl (records) + data/faces/ (compressed thumbnails).
Kept off git via .gitignore — uploads stay local to the edge device.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "face_db.jsonl"
FACES_DIR = BASE / "data" / "faces"


def load_db() -> list[dict]:
    """Load the demo face database (records only, no image bytes)."""
    if not DB_PATH.exists():
        return []
    try:
        out = []
        with open(DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    except Exception:
        return []


def _save_db(records: list[dict]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _thumb_path(fid: str) -> Path:
    return FACES_DIR / f"{fid}.jpg"


def _store_thumb(fid: str, img_bytes: bytes) -> bool:
    """Downscale + re-encode the uploaded image and store locally."""
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        scale = 220 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(_thumb_path(fid)), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return True
    except Exception:
        return False


def register_face(name: str, note: str, img_bytes: bytes) -> dict | None:
    """Register one person in the demo face database. Returns the record."""
    name = (name or "").strip()
    if not name or not img_bytes:
        return None
    records = load_db()
    fid = f"f{int(time.time() * 1000)}{len(records)}"
    if not _store_thumb(fid, img_bytes):
        return None
    rec = {
        "id": fid,
        "name": name,
        "note": (note or "").strip(),
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    records.append(rec)
    _save_db(records)
    return rec


def find_by_name(q: str) -> list[dict]:
    q = (q or "").strip().lower()
    if not q:
        return []
    return [r for r in load_db() if q in r["name"].lower()]


def remove_person(pid: str) -> bool:
    records = [r for r in load_db() if r["id"] != pid]
    _save_db(records)
    _thumb_path(pid).unlink(missing_ok=True)
    return True


def render_face_intelligence(make_alert) -> None:
    """Full Face Intelligence section (DEMO-labelled)."""
    import streamlit as st

    st.subheader("🧑 Face Intelligence")
    st.markdown(
        "<span class='chip chip-sim'>● SIMULATED FACE DATABASE</span> "
        "<span class='chip chip-info'>● DETECTION ≠ IDENTIFICATION</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Face *detection* (Haar, built into the engine) counts faces live. "
        "Identity lookup below is a **simulated face database** — no real "
        "biometric matching is claimed."
    )

    tab_reg, tab_search, tab_status = st.tabs(
        ["📥 Register Person", "🔎 Search Person", "🎥 Footage Status"])

    with tab_reg:
        img = st.file_uploader("Face image (jpg/png)", type=["jpg", "jpeg", "png"],
                               key="face_reg_img")
        name = st.text_input("Person name", key="face_reg_name",
                             placeholder="e.g. Unknown Subject A")
        note = st.text_area("Short note / description", key="face_reg_note",
                            height=80)
        if st.button("➕ Add to Face Database", key="face_reg_go",
                     type="primary", use_container_width=True):
            if not img:
                st.warning("Upload a face image first.")
            elif not (name or "").strip():
                st.warning("Enter a person name first.")
            else:
                rec = register_face(name, note, img.getvalue())
                if rec:
                    st.success(
                        f"Registered **{rec['name']}** (simulated DB) — "
                        f"ID `{rec['id']}`")
                    make_alert(
                        "face_registered",
                        f"Operator registered person '{rec['name']}' "
                        "in the face database.",
                        "low", payload={"face_id": rec["id"]})
                    st.rerun()
                else:
                    st.error("Could not read the uploaded image.")

    with tab_search:
        q = st.text_input("Search by name", key="face_search_q")
        c1, c2 = st.columns([3, 1])
        hit_btn = c1.button("🔎 Search Database", key="face_search_go",
                            use_container_width=True)
        if c2.button("🗑 Clear", key="face_search_clear"):
            st.session_state.face_search_res = None
        if hit_btn or st.session_state.get("face_search_res") is not None:
            if hit_btn:
                st.session_state.face_search_res = (q, find_by_name(q))
            _q, hits = st.session_state.face_search_res
            if hits:
                st.success(f"{len(hits)} profile(s) found for '{_q}' "
                           "(simulated lookup — name match only).")
                for r in hits:
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        thumb = _thumb_path(r["id"])
                        if thumb.exists():
                            st.image(str(thumb), width=120)
                    with col_b:
                        st.markdown(f"**{r['name']}**")
                        st.caption(f"Registered: {r['registered_at']} · ID `{r['id']}`")
                        st.caption(r["note"] or "—")
                        st.markdown(
                            "<span class='chip chip-live'>● REGISTERED ✓ "
                            "Profile found (SIMULATED)</span>",
                            unsafe_allow_html=True)
                        if st.button("❌ Remove", key=f"face_del_{r['id']}"):
                            remove_person(r["id"])
                            st.rerun()
            else:
                st.warning(
                    f"No profile for '{_q}' — **Not Found** in face database. "
                    "Register them first, or check the footage status tab.")
                st.markdown(
                    "<span class='chip chip-off'>● NOT FOUND (SIMULATED)</span>",
                    unsafe_allow_html=True)
        st.divider()
        st.caption(
            "Upload a face to search: this build does NOT compare "
            "facial embeddings. Uploaded faces are shown for reference only "
            "and never claimed as a biometric identity match.")

    with tab_status:
        last = st.session_state.get("last_faces", 0)
        st.metric("Faces detected in current footage", last)
        seen = st.session_state.get("face_seen", False)
        if seen:
            st.markdown(
                "<span class='pill pill-live'>PREVIOUSLY SEEN ✓ "
                "(simulated session flag)</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<span class='pill pill-off'>NOT SEEN IN CURRENT FOOTAGE "
                "(simulated)</span>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("👁 Mark as seen now", key="face_seen_on",
                     use_container_width=True):
            st.session_state.face_seen = True
            make_alert("face_detected",
                       "Operator marked a face as seen in current footage "
                       "(simulated).",
                       "low", payload={"status": "seen"})
            st.rerun()
        if c2.button("↩ Reset status", key="face_seen_off",
                     use_container_width=True):
            st.session_state.face_seen = False
            st.rerun()
        st.caption(
            "'Previously seen' is a manual session flag for simulation "
            "purposes — it is not a real re-identification result.")