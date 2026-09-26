"""Border Map — camera wall → coverage → virtual fence → alerts.

Two render paths:
1. Interactive Folium map (used when streamlit-folium is installed and the
   user has connectivity to fetch tiles).
2. Offline canvas map built with OpenCV/numpy — always available, clearly
   labelled DEMO BORDER MAP with simulated coordinates.

Nothing here ever claims the coordinates are a real geographic position.
"""
from __future__ import annotations

import numpy as np

# Simulated sector geometry (units are map pixels, not real metres).
CAMERAS = {
    "CAM-01": {"name": "North Gate",   "pos": (150, 150), "dir": (0, -60)},
    "CAM-02": {"name": "Wire Zone",    "pos": (320, 90),  "dir": (60, -30)},
    "CAM-03": {"name": "Checkpoint",   "pos": (480, 260), "dir": (0, 55)},
    "CAM-04": {"name": "Riverine",     "pos": (90, 360),  "dir": (70, 0)},
}
FENCE = [(200, 200), (430, 200), (430, 330), (200, 330)]
RESTRICTED = [(260, 240), (370, 240), (370, 300), (260, 300)]


def _real_coords() -> dict:
    """Real camera lat/lng from session map_coords (sites.json).

    Empty dict means 'no real coordinates supplied' — the map then falls back
    to the illustrative layout below and says so.
    """
    import streamlit as st
    mc = st.session_state.get("map_coords", {}) or {}
    out = {}
    for cid in CAMERAS:
        ent = mc.get(cid)
        if isinstance(ent, dict):
            try:
                out[cid] = (float(ent.get("lat")), float(ent.get("lng")))
            except (TypeError, ValueError):
                pass
    return out


def render_border_map() -> None:
    """Show the mapped implementation with sim-live session data."""
    import streamlit as st

    st.subheader("🗺 Border Map")
    use_folium = st.checkbox(
        "Interactive Folium map (needs tile connectivity)",
        value=True, key="map_folium",
        help="Turn OFF for the offline canvas map — works with no internet.")

    cam_status = st.session_state.get("camera_status", {})
    alerts = st.session_state.get("alerts", [])
    real = _real_coords()
    if real:
        st.markdown(
            "<span class='chip chip-live'>● REAL COORDINATES LOADED</span> "
            f"<span class='tech'>Cameras: {', '.join(sorted(real))} — "
            f"plotted from sites.json, not simulated.</span>",
            unsafe_allow_html=True)
    if use_folium and _try_folium(cam_status, alerts):
        pass
    else:
        st.markdown(
            "<span class='chip chip-sim'>● SIMULATED BORDER MAP — "
            "ILLUSTRATIVE COORDINATES</span>", unsafe_allow_html=True)
        st.caption(
            "Layout: CAMERA → COVERAGE AREA → VIRTUAL FENCE → ALERT. "
            "Positions are illustrative, not real survey coordinates."
            + (" Real lat/lng from sites.json apply to the Folium map — turn "
               "it ON to see the real positions." if real else ""))
        img = _canvas_map(cam_status, alerts)
        st.image(img, channels="BGR", use_container_width=True)
        _canvas_legend()


def _try_folium(cam_status: dict, alerts: list) -> bool:
    """Render the Folium path. Returns True on success, False on any error."""
    import streamlit as st
    try:
        import folium
        from modules.checks import HAS_FOLIUM
        if not HAS_FOLIUM:
            raise RuntimeError("folium/streamlit-folium not installed")
        from streamlit_folium import st_folium

        center = [28.6245, 77.0827]  # simulated border-sector midpoint
        m = folium.Map(location=center, zoom_start=14,
                       tiles="CartoDB positron", control_scale=True)

        # Virtual fence boundary + restricted zone
        fence_latlng = _to_latlng(FENCE, center)
        folium.PolyLine(fence_latlng, color="#16A34A", weight=3,
                        opacity=0.9, tooltip="Virtual Fence").add_to(m)
        folium.Polygon(_to_latlng(RESTRICTED, center),
                       color="#D97706", weight=2, fill=True,
                       fill_opacity=0.12,
                       tooltip="Restricted Zone").add_to(m)
        folium.Marker(_to_latlng((320, 265), center)[0],
                      icon=folium.DivIcon(html='<b style="color:#D97706">RESTRICTED</b>')
                      ).add_to(m)

        # Cameras: green online / red offline, with coverage insets
        real = _real_coords()

        def _ll(cid: str):
            """Real lat/lng when supplied, otherwise the illustrative offset."""
            if cid in real:
                return real[cid]
            cfg = CAMERAS.get(cid, CAMERAS["CAM-01"])
            return _to_latlng(cfg["pos"], center)[0]

        for cid, cfg in CAMERAS.items():
            lat0, lng0 = _ll(cid)
            ok = cam_status.get(cid, True)
            colour = "#16A34A" if ok else "#DC2626"
            _where = (f"@{lat0:.5f},{lng0:.5f}" if cid in real
                      else "illustrative position")
            folium.Marker(
                [lat0, lng0], tooltip=f"{cid} — {cfg['name']} "
                                      f"({'ONLINE' if ok else 'OFFLINE'}) · {_where}",
                icon=folium.Icon(color="green" if ok else "red",
                                 icon="video-camera", prefix="fa")
            ).add_to(m)
            folium.Circle([lat0, lng0], radius=90, color=colour,
                          weight=1.5, fill=True, fill_opacity=0.10,
                          popup=f"{cid} coverage").add_to(m)

        # Recent alerts → markers coloured by severity
        sev_col = {"critical": "red", "high": "orange", "medium": "beige",
                   "low": "lightblue"}
        for a in alerts[-10:]:
            cam = a.get("camera_id", "CAM-01")
            if cam not in CAMERAS:
                continue
            lat0, lng0 = _ll(cam)
            folium.Marker(
                [lat0, lng0],
                popup=f"{a['event_type'].upper()} — {a.get('explanation','')[:80]}",
                icon=folium.Icon(color=sev_col.get(a.get("severity"), "gray"),
                                 icon="exclamation-triangle", prefix="fa"),
            ).add_to(m)

        st_folium(m, width="100%", height=500)
        return True
    except Exception:
        return False


def _to_latlng(pt, center):
    """Scale map-pixel coords into a small lat/lng box around the centre."""
    lat, lng = center
    x, y = pt
    dx = (x - 320) * 0.00042
    dy = (y - 265) * 0.00042
    return [(lat + dy, lng + dx)]


def _canvas_map(cam_status: dict, alerts: list) -> np.ndarray:
    """Offline OpenCV map: border sector, fence, cameras, coverage, alerts."""
    import cv2

    W, H = 920, 520
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (236, 238, 240)          # terrain base
    # grid
    for g in range(0, W, 60):
        cv2.line(img, (g, 0), (g, H), (214, 220, 224), 1)
    for g in range(0, H, 60):
        cv2.line(img, (0, g), (W, g), (214, 220, 224), 1)
    cv2.putText(img, "SIMULATED SECTOR — NOT REAL COORDINATES", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1)

    # restricted zone
    cv2.fillPoly(img, [np.array(RESTRICTED + [(470, 300)], np.int32)],
                 (214, 214, 120))
    cv2.polylines(img, [np.array([(260, 240), (370, 240), (370, 300),
                                   (260, 300)], np.int32)],
                  True, (40, 140, 180), 2)

    # virtual fence
    cv2.polylines(img, [np.array(FENCE, np.int32)], True, (80, 190, 120), 3)
    cv2.putText(img, "VIRTUAL FENCE", (215, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 140, 90), 1)

    # border line (bottom strip)
    cv2.line(img, (30, 470), (W - 30, 470), (120, 120, 130), 3)
    cv2.putText(img, "BORDER LINE", (40, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 100), 1)

    for cid, cfg in CAMERAS.items():
        ok = cam_status.get(cid, True)
        col = (80, 190, 110) if ok else (90, 90, 230)
        x, y = cfg["pos"]
        cv2.circle(img, (int(x * 1.6), int(y * 1.05)), 8, col, -1)
        cv2.circle(img, (int(x * 1.6), int(y * 1.05)), 26, col, 2)
        cv2.putText(img, cid, (int(x * 1.6) + 32, int(y * 1.05) - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(img, cfg["name"], (int(x * 1.6) + 32, int(y * 1.05) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1)
        t = cv2.FONT_HERSHEY_SIMPLEX
        cv2.line(img, (int(x * 1.6), int(y * 1.05)),
                 (int(x * 1.6) + cfg["dir"][0], int(y * 1.05) + cfg["dir"][1]),
                 col, 1, cv2.LINE_AA)

    sev_col = {"critical": (60, 60, 230), "high": (50, 140, 210),
               "medium": (60, 190, 240), "low": (130, 180, 110)}
    for a in alerts[-8:]:
        cam = a.get("camera_id", "CAM-01")
        cfg = CAMERAS.get(cam, CAMERAS["CAM-01"])
        x, y = cfg["pos"]
        col = sev_col.get(a.get("severity"), (120, 120, 120))
        cv2.drawMarker(img, (int(x * 1.6), int(y * 1.05) + 34), col,
                       cv2.MARKER_CROSS, 14, 2)
    return img


def _canvas_legend() -> None:
    import streamlit as st
    st.markdown(
        "<div class='map-note'>"
        "🟢 Online camera &nbsp;•&nbsp; 🔴 Offline camera &nbsp;•&nbsp; "
        "<span style='color:#16A34A'>■</span> Virtual fence &nbsp;•&nbsp; "
        "<span style='color:#D97706'>■</span> Restricted zone &nbsp;•&nbsp; "
        "✕ Alert marker (colour = severity)"
        "</div>", unsafe_allow_html=True)