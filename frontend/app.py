import streamlit as st
import requests
import base64
import json
import math
from datetime import datetime

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Debris Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API = "http://127.0.0.1:8000"
HISTORY_KEY = "debris_history"

# ─── Theme / CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global background ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(135deg, #020617 0%, #0f172a 60%, #020c1b 100%) !important;
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0f172a !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #0ea5e9; border-radius: 3px; }

/* ── glass card ── */
.glass {
    background: rgba(15,23,42,0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(14,165,233,0.12);
    border-radius: 16px;
    padding: 24px;
}
.glow { box-shadow: 0 0 24px rgba(14,165,233,0.2); }

/* ── stat card ── */
.stat-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(14,165,233,0.12);
    border-radius: 16px;
    padding: 20px;
    display: flex;
    align-items: flex-start;
    gap: 16px;
}
.stat-icon { font-size: 2rem; }
.stat-label { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.stat-value { font-size: 1.5rem; font-weight: 700; color: white; }
.stat-sub { color: #94a3b8; font-size: 11px; margin-top: 4px; }

/* ── badge ── */
.badge {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 9999px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: white;
}

/* ── image card ── */
.img-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(14,165,233,0.12);
    border-radius: 16px;
    overflow: hidden;
}
.img-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.img-card-title { font-weight: 600; font-size: 14px; color: #e2e8f0; }
.img-tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 9999px;
    font-weight: 600;
    background: rgba(14,165,233,0.2);
    color: #38bdf8;
    border: 1px solid rgba(14,165,233,0.3);
}

/* ── history row ── */
.history-row {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    transition: background 0.15s;
}
.history-row:hover { background: rgba(255,255,255,0.02); }
.history-icon { font-size: 1.5rem; }
.history-name { font-weight: 500; font-size: 14px; }
.history-time { color: #94a3b8; font-size: 12px; margin-top: 2px; }
.history-count { color: white; font-weight: 700; font-size: 16px; text-align: right; }
.history-count-label { color: #94a3b8; font-size: 11px; }

/* ── progress bar ── */
.progress-wrap { height: 8px; border-radius: 9999px; background: #1e293b; overflow: hidden; margin-top: 4px; }
.progress-fill { height: 100%; border-radius: 9999px; }

/* ── tab bar ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(14,165,233,0.12);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 8px;
    font-weight: 500;
    font-size: 14px;
    padding: 8px 16px;
    border: none;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
    color: white !important;
    box-shadow: 0 0 16px rgba(14,165,233,0.4);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* ── upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.85) !important;
    border: 2px dashed rgba(14,165,233,0.25) !important;
    border-radius: 16px !important;
    color: #94a3b8 !important;
}
[data-testid="stFileUploader"] label { color: #94a3b8 !important; }

/* ── run button ── */
.stButton > button {
    background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 32px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: 0 0 20px rgba(14,165,233,0.35) !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
.stButton > button:disabled { opacity: 0.5 !important; }

/* ── metric ── */
[data-testid="stMetric"] { background: transparent !important; }

/* ── section header ── */
.section-header {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
}

/* ── alert boxes ── */
.alert-error {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 16px;
    padding: 20px;
    color: #fca5a5;
    font-size: 14px;
}
.alert-info {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 16px;
    padding: 20px;
    color: #fcd34d;
    font-size: 14px;
}
.alert-tip {
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 12px;
    padding: 16px;
    color: #7dd3fc;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if HISTORY_KEY not in st.session_state:
    st.session_state[HISTORY_KEY] = []

# ─── Helpers ──────────────────────────────────────────────────────────────────
BADGE_COLORS = ["#06b6d4","#10b981","#8b5cf6","#f59e0b","#ef4444","#14b8a6","#6366f1"]
RING_COLORS  = ["#0ea5e9","#10b981","#8b5cf6","#f59e0b","#ef4444","#14b8a6","#6366f1"]

def badge_color(name: str) -> str:
    return BADGE_COLORS[ord(name[0]) % len(BADGE_COLORS)] if name else BADGE_COLORS[0]

def badge_html(name: str) -> str:
    color = badge_color(name)
    return f'<span class="badge" style="background:{color}">{name}</span>'

def progress_ring_svg(pct: int, label: str, color: str = "#0ea5e9") -> str:
    r = 36
    c = 2 * math.pi * r
    offset = c - (pct / 100) * c
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px">
      <svg width="90" height="90" viewBox="0 0 90 90">
        <circle cx="45" cy="45" r="{r}" fill="none" stroke="#1e293b" stroke-width="7"/>
        <circle cx="45" cy="45" r="{r}" fill="none" stroke="{color}" stroke-width="7"
          stroke-dasharray="{c:.2f}" stroke-dashoffset="{offset:.2f}" stroke-linecap="round"
          style="transform:rotate(-90deg);transform-origin:center;transition:stroke-dashoffset 0.6s ease"/>
        <text x="45" y="50" text-anchor="middle" fill="white" font-size="14" font-weight="700">{pct}%</text>
      </svg>
      <span style="font-size:11px;color:#94a3b8;text-align:center;max-width:80px">{label}</span>
    </div>"""

def stat_card_html(icon: str, label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="stat-card">
      <div class="stat-icon">{icon}</div>
      <div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {sub_html}
      </div>
    </div>"""

def img_card_html(title: str, b64: str, tag: str = "", mime: str = "jpeg") -> str:
    tag_html = f'<span class="img-tag">{tag}</span>' if tag else ""
    return f"""
    <div class="img-card">
      <div class="img-card-header">
        <span class="img-card-title">{title}</span>
        {tag_html}
      </div>
      <img src="data:image/{mime};base64,{b64}" style="width:100%;display:block"/>
    </div>"""

# ─── Navbar ───────────────────────────────────────────────────────────────────
history = st.session_state[HISTORY_KEY]
count_badge = f' <span style="background:rgba(14,165,233,0.3);color:#38bdf8;font-size:12px;padding:1px 8px;border-radius:9999px">{len(history)}</span>' if history else ""

st.markdown(f"""
<div style="background:rgba(15,23,42,0.85);backdrop-filter:blur(16px);
     border-bottom:1px solid rgba(14,165,233,0.1);
     padding:16px 32px;display:flex;align-items:center;justify-content:space-between;
     margin-bottom:32px;border-radius:0 0 16px 16px">
  <div style="display:flex;align-items:center;gap:12px">
    <span style="font-size:2rem">🌊</span>
    <div>
      <div style="font-weight:700;font-size:18px;line-height:1.2">Debris Intelligence</div>
      <div style="color:#94a3b8;font-size:12px">AI-powered underwater debris detection &amp; classification</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_dashboard, tab_analyze = st.tabs([f"📊 Dashboard{count_badge}", "🔬 Analyze"])

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    if not history:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
             padding:128px 0;gap:16px">
          <div style="font-size:4rem">🌊</div>
          <p style="color:#94a3b8;font-size:18px">No analyses yet. Run your first detection in the Analyze tab.</p>
        </div>""", unsafe_allow_html=True)
    else:
        # ── aggregate stats ──
        total_scans   = len(history)
        total_objects = sum(h.get("total_objects") or 0 for h in history)
        conf_vals     = [h["average_confidence"] for h in history if h.get("average_confidence")]
        avg_conf      = sum(conf_vals) / len(conf_vals) if conf_vals else None

        class_agg: dict[str, int] = {}
        for h in history:
            for k, v in (h.get("class_counts") or {}).items():
                class_agg[k] = class_agg.get(k, 0) + v

        top_class = max(class_agg.items(), key=lambda x: x[1]) if class_agg else None
        total_for_pct = sum(class_agg.values())

        # stat strip
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(stat_card_html("🔍", "Total Scans", str(total_scans)), unsafe_allow_html=True)
        with c2:
            st.markdown(stat_card_html("🗑️", "Objects Found", str(total_objects), "across all sessions"), unsafe_allow_html=True)
        with c3:
            conf_str = f"{avg_conf*100:.1f}%" if avg_conf is not None else "—"
            st.markdown(stat_card_html("📊", "Avg Confidence", conf_str), unsafe_allow_html=True)
        with c4:
            top_val = top_class[0] if top_class else "—"
            top_sub = f"{top_class[1]} detections" if top_class else ""
            st.markdown(stat_card_html("🏆", "Top Debris Type", top_val, top_sub), unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── class distribution rings ──
        if class_agg:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Class Distribution (All Sessions)</div>', unsafe_allow_html=True)
            ring_html = '<div style="display:flex;flex-wrap:wrap;gap:24px;justify-content:center">'
            for i, (cls, count) in enumerate(class_agg.items()):
                pct = round(count / total_for_pct * 100)
                ring_html += progress_ring_svg(pct, cls, RING_COLORS[i % len(RING_COLORS)])
            ring_html += "</div>"
            st.markdown(ring_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── session history ──
        col_left, col_right = st.columns([5, 1])
        with col_left:
            st.markdown('<div class="section-header">Session History</div>', unsafe_allow_html=True)
        with col_right:
            if st.button("Clear All", key="clear_history"):
                st.session_state[HISTORY_KEY] = []
                st.rerun()

        rows_html = '<div class="glass" style="padding:0;overflow:hidden">'
        for h in reversed(history):
            icon = "🎬" if h.get("type") == "video" else "🖼️"
            badges = "".join(badge_html(cls) for cls in (h.get("class_counts") or {}).keys())
            count_val = h.get("total_objects") if h.get("total_objects") is not None else "—"
            rows_html += f"""
            <div class="history-row">
              <span class="history-icon">{icon}</span>
              <div style="flex:1;min-width:0">
                <div class="history-name" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                  {h.get('filename','—')}
                </div>
                <div class="history-time">{h.get('timestamp','')}</div>
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end">{badges}</div>
              <div style="text-align:right;flex-shrink:0;min-width:48px">
                <div class="history-count">{count_val}</div>
                <div class="history-count-label">objects</div>
              </div>
            </div>"""
        rows_html += "</div>"
        st.markdown(rows_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    uploaded = st.file_uploader(
        "Drag & drop or click to upload an **image** or **video**",
        type=["jpg","jpeg","png","mp4","avi","mov","mkv"],
        help="Supported: jpg, png, mp4, avi, mov, mkv",
        label_visibility="visible",
    )

    if uploaded:
        is_video = uploaded.type.startswith("video")
        st.markdown(
            f'<div style="color:#38bdf8;font-weight:600;margin:8px 0">{"🎬" if is_video else "🖼️"} {uploaded.name}</div>',
            unsafe_allow_html=True,
        )

        col_btn, _ = st.columns([1, 4])
        with col_btn:
            run_clicked = st.button("Run Detection", use_container_width=True)

        if run_clicked:
            endpoint = f"{API}/detect-video-with-heatmap/" if is_video else f"{API}/detect-image/"
            label_txt = "Processing video frames + tracking…" if is_video else "Running YOLOv8 inference…"
            detail_txt = (
                "Video processing may take a minute. CLAHE enhancement + DeepSORT tracking running…"
                if is_video else
                "CLAHE enhancement + object classification in progress…"
            )

            progress_bar = st.progress(0, text=label_txt)
            status_txt   = st.empty()
            status_txt.markdown(
                f'<div style="color:#64748b;font-size:12px;text-align:center">{detail_txt}</div>',
                unsafe_allow_html=True,
            )

            # animate progress while waiting
            import threading, time

            stop_flag = threading.Event()
            prog_val  = [0]

            def tick():
                step = 1.2 if is_video else 4
                while not stop_flag.is_set():
                    prog_val[0] = min(prog_val[0] + step, 90)
                    time.sleep(0.2)

            t = threading.Thread(target=tick, daemon=True)
            t.start()

            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(endpoint, files=files, timeout=600)
                stop_flag.set()
                progress_bar.progress(100, text="Done!")
                status_txt.empty()
                data = response.json()

                # save to history
                st.session_state[HISTORY_KEY].append({
                    "filename":          uploaded.name,
                    "type":              "video" if is_video else "image",
                    "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_objects":     data.get("total_objects"),
                    "average_confidence":data.get("average_confidence"),
                    "class_counts":      data.get("class_counts", {}),
                })

                # ── IMAGE RESULTS ──
                if not is_video:
                    orig_b64 = data.get("original_base64","")
                    enh_b64  = data.get("enhanced_base64","")
                    det_b64  = data.get("detected_base64","")
                    class_counts = data.get("class_counts", {})
                    class_pcts   = data.get("class_percentages", {})
                    total_objs   = data.get("total_objects", 0)
                    avg_conf     = data.get("average_confidence", 0)

                    # 3-panel image strip
                    ic1, ic2, ic3 = st.columns(3)
                    with ic1:
                        st.markdown(img_card_html("Original", orig_b64, "RAW"), unsafe_allow_html=True)
                    with ic2:
                        st.markdown(img_card_html("Enhanced (CLAHE)", enh_b64, "ENHANCED"), unsafe_allow_html=True)
                    with ic3:
                        st.markdown(img_card_html("Detection Output", det_b64, "DETECTED"), unsafe_allow_html=True)

                    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                    # analytics row
                    left_col, right_col = st.columns(2)

                    # class breakdown bars
                    with left_col:
                        st.markdown('<div class="glass">', unsafe_allow_html=True)
                        st.markdown('<div class="section-header">Class Breakdown</div>', unsafe_allow_html=True)
                        if class_counts:
                            total_for_ring = sum(class_counts.values())
                            bars_html = ""
                            for i, (cls, count) in enumerate(class_counts.items()):
                                pct = class_pcts.get(cls, 0)
                                color = RING_COLORS[i % len(RING_COLORS)]
                                bars_html += f"""
                                <div style="margin-bottom:12px">
                                  <div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:4px">
                                    <span style="display:flex;align-items:center;gap:8px">
                                      {badge_html(cls)} <span>{cls}</span>
                                    </span>
                                    <span style="color:#cbd5e1">{count} &nbsp;·&nbsp; {pct}%</span>
                                  </div>
                                  <div class="progress-wrap">
                                    <div class="progress-fill" style="width:{pct}%;background:{color}"></div>
                                  </div>
                                </div>"""
                            st.markdown(bars_html, unsafe_allow_html=True)
                        else:
                            st.markdown('<p style="color:#64748b;font-size:14px">No detections.</p>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # summary rings + totals
                    with right_col:
                        st.markdown('<div class="glass">', unsafe_allow_html=True)
                        st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
                        if class_counts:
                            total_for_ring = sum(class_counts.values())
                            rings_html = '<div style="display:flex;flex-wrap:wrap;gap:24px;justify-content:center">'
                            for i, (cls, count) in enumerate(class_counts.items()):
                                pct = round(count / total_for_ring * 100)
                                rings_html += progress_ring_svg(pct, cls, RING_COLORS[i % len(RING_COLORS)])
                            rings_html += "</div>"
                            st.markdown(rings_html, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="border-top:1px solid rgba(255,255,255,0.05);margin-top:16px;padding-top:16px">
                          <div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:8px">
                            <span style="color:#94a3b8">Total Objects</span>
                            <span style="font-weight:700;color:white">{total_objs}</span>
                          </div>
                          <div style="display:flex;justify-content:space-between;font-size:14px">
                            <span style="color:#94a3b8">Avg Confidence</span>
                            <span style="font-weight:700;color:#38bdf8">{(avg_conf or 0)*100:.1f}%</span>
                          </div>
                        </div>""", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                # ── VIDEO RESULTS ──
                else:
                    message        = data.get("message")
                    heatmap_b64    = data.get("heatmap_image_base64","")
                    output_video   = data.get("output_video","")
                    heatmap_csv    = data.get("heatmap_csv","")

                    if message:
                        st.markdown(f'<div class="alert-info">ℹ️ {message}</div>', unsafe_allow_html=True)
                        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                    if heatmap_b64:
                        vc1, vc2 = st.columns(2)
                        with vc1:
                            st.markdown(img_card_html("Spatial Debris Heatmap", heatmap_b64, "HEATMAP", "png"), unsafe_allow_html=True)
                        with vc2:
                            vid_file = output_video.replace("\\","/").split("/")[-1] if output_video else "—"
                            csv_file = heatmap_csv.replace("\\","/").split("/")[-1] if heatmap_csv else ""
                            csv_row  = f"""
                            <div style="display:flex;justify-content:space-between;font-size:14px;margin-top:8px">
                              <span style="color:#94a3b8">CSV Export</span>
                              <span style="color:#38bdf8;font-size:12px;overflow:hidden;text-overflow:ellipsis;max-width:200px">{csv_file}</span>
                            </div>""" if csv_file else ""
                            st.markdown(f"""
                            <div class="glass">
                              <div class="section-header">Video Analysis</div>
                              <p style="color:#94a3b8;font-size:14px">
                                The heatmap shows where debris was detected most frequently across all video frames.
                                Brighter spots indicate higher debris concentration.
                              </p>
                              <div style="border-top:1px solid rgba(255,255,255,0.05);margin-top:16px;padding-top:16px">
                                <div style="display:flex;justify-content:space-between;font-size:14px">
                                  <span style="color:#94a3b8">Output Video</span>
                                  <span style="color:#38bdf8;font-size:12px;overflow:hidden;text-overflow:ellipsis;max-width:200px">{vid_file}</span>
                                </div>
                                {csv_row}
                              </div>
                              <div class="alert-tip" style="margin-top:16px">
                                💡 Find the annotated video in <code style="color:#38bdf8">backend/outputs/</code>
                              </div>
                            </div>""", unsafe_allow_html=True)
                    elif output_video:
                        st.markdown(f"""
                        <div class="glass" style="font-size:14px;color:#94a3b8">
                          Video saved to: <code style="color:#38bdf8">{output_video}</code>
                        </div>""", unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                stop_flag.set()
                progress_bar.empty()
                status_txt.empty()
                st.markdown("""
                <div class="alert-error">
                  ⚠️ Could not reach the backend. Make sure uvicorn is running on port 8000.
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                stop_flag.set()
                progress_bar.empty()
                status_txt.empty()
                st.markdown(f'<div class="alert-error">⚠️ {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
             padding:64px 0;gap:12px">
          <div style="font-size:3rem">☁️</div>
          <p style="color:#94a3b8">Drag &amp; drop or click above to upload an <strong>image</strong> or <strong>video</strong></p>
          <p style="color:#64748b;font-size:12px">Supported: jpg, png, mp4, avi, mov, mkv</p>
        </div>""", unsafe_allow_html=True)
