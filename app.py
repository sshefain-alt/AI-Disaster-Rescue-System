"""
Web Interface Module - AI Disaster Rescue System
Realistic map visualization with Google Maps style route tracking.

Run with:  python -m streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from environment import Environment
from search import (bfs, dfs, astar, risk_aware_astar,
                    greedy_best_first, hill_climbing)
from csp import CSPAmbulanceAllocator
from ml_model import RescueMLModel
from fuzzy import FuzzyRescueDecision
from agent import RescueAgent


# ============================================================================
# PLOTLY VERSION COMPATIBILITY
# plotly < 6.0 uses Scattermapbox / layout.mapbox
# plotly >= 6.0 uses Scattermap     / layout.map
# ============================================================================
def _uses_scattermap():
    return hasattr(go, "Scattermap")


def _scatter_cls():
    """Return the correct scatter trace class for the installed plotly."""
    if _uses_scattermap():
        return getattr(go, "Scattermap")
    return go.Scattermapbox


def _kind():
    return "scattermap" if _uses_scattermap() else "scattermapbox"


def _add_pin(fig, lat, lon, color, symbol, text, size, hovertext,
             name, showlegend=True, textfont_size=11):
    """
    Draw a map pin with a white halo.

    Scattermap markers are maki icons and cannot carry a stroke, so the
    white ring is faked with a slightly larger white circle underneath.
    """
    S = _scatter_cls()
    fig.add_trace(S(
        type=_kind(), mode="markers",
        lon=[lon], lat=[lat],
        marker=dict(size=size + 6, color="white", symbol="circle",
                    opacity=0.95),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(S(
        type=_kind(), mode="markers+text",
        lon=[lon], lat=[lat],
        marker=dict(size=size, color=color, symbol=symbol),
        text=[text] if text else None,
        textposition="middle center",
        textfont=dict(size=textfont_size, color="white"),
        hovertext=hovertext, hoverinfo="text",
        name=name, showlegend=showlegend
    ))


def _map_layout_kwargs(style):
    """Return map layout kwargs for the installed plotly version."""
    if _uses_scattermap():
        return {"map": dict(style=style)}
    return {"mapbox": dict(style=style)}


# Google / Material colours
GOOGLE_BLUE = "#1a73e8"
ROUTE_CASING = "#ffffff"
BUILDING_FILL = "#e6e2dc"
BUILDING_EDGE = "#d8d3ca"
ROAD_COLOR = "#ffffff"
ROAD_EDGE = "#d9d4cc"

MAP_STYLES = {
    "Clean (Google-like)": "carto-positron",
    "Streets": "open-street-map",
    "Dark": "carto-darkmatter",
}

# Scale: 1 grid cell = 100 metres
CELL_KM = 0.1
AVG_SPEED_KMH = 25.0


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================
def to_latlon(row, col, center_lat=40.7128, center_lon=-74.0060, scale=0.0012):
    """Convert a grid coordinate into geographic lat/lon."""
    return (center_lat - row * scale, center_lon + col * scale)


def bearing_between(a, b):
    """Return compass heading name from grid point a to grid point b."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr < 0:
        return "North"
    if dr > 0:
        return "South"
    if dc > 0:
        return "East"
    return "West"


def get_turn_by_turn(path):
    """Build a list of Google-style navigation instructions from a path."""
    if not path or len(path) < 2:
        return []
    order = ["North", "East", "South", "West"]
    turn_text = {0: "Continue straight", 1: "Turn right",
                 2: "Make a U-turn", 3: "Turn left"}

    bearings = [bearing_between(path[i], path[i + 1])
                for i in range(len(path) - 1)]
    steps = []
    for i, b in enumerate(bearings):
        if i == 0:
            steps.append({"icon": "\U0001F6EB", "text": f"Head {b.lower()}"})
            continue
        prev = bearings[i - 1]
        if b == prev:
            continue
        diff = (order.index(b) - order.index(prev)) % 4
        steps.append({"icon": "\U0001F503",
                      "text": f"{turn_text[diff]} onto {b}"})
    return steps


def path_metrics(path):
    """Distance / duration / ETA for a path, Google style."""
    steps = max(len(path) - 1, 0)
    km = steps * CELL_KM
    minutes = (km / AVG_SPEED_KMH) * 60 if km else 0
    return {
        "steps": steps,
        "km": km,
        "meters": km * 1000,
        "minutes": minutes,
        "duration": (f"{int(minutes)} min" if minutes >= 1 else "< 1 min"),
        "eta": f"{int(minutes)} min" if minutes else "now",
    }


# ============================================================================
# MAP LAYERS
# ============================================================================
def _add_buildings(fig, env, path_cells):
    """Draw city blocks so the map reads like a real urban area."""
    S = _scatter_cls()
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in path_cells:
                continue
            if env.grid[r][c] in (env.HOSPITAL, env.VICTIM):
                continue  # keep landmark tiles open

            lat, lon = to_latlon(r, c)
            d = 0.00042
            fig.add_trace(S(
                type="scattermap" if _uses_scattermap() else "scattermapbox",
                mode="lines",
                lon=[lon - d, lon + d, lon + d, lon - d, lon - d],
                lat=[lat + d, lat + d, lat - d, lat - d, lat + d],
                line=dict(color=BUILDING_EDGE, width=1),
                fill="toself",
                fillcolor=BUILDING_FILL,
                hoverinfo="skip",
                showlegend=False
            ))


def _add_road_network(fig, env, path_cells):
    """Draw the drivable street network, highlighting the active route."""
    S = _scatter_cls()
    lats, lons = [], []

    for r in range(env.rows):
        row_lats, row_lons = [], []
        for c in range(env.cols):
            lat, lon = to_latlon(r, c)
            row_lats.append(lat)
            row_lons.append(lon)
            if c < env.cols - 1 and env.grid[r][c + 1] != env.BLOCKED:
                lats.append(lat)
                lons.append(lon)
        if row_lats:
            lats.extend(row_lats + [None])
            lons.extend(row_lons + [None])

    for c in range(env.cols):
        col_lats, col_lons = [], []
        for r in range(env.rows):
            lat, lon = to_latlon(r, c)
            col_lats.append(lat)
            col_lons.append(lon)
            if r < env.rows - 1 and env.grid[r + 1][c] != env.BLOCKED:
                lats.append(lat)
                lons.append(lon)
        if col_lats:
            lats.extend(col_lats + [None])
            lons.extend(col_lons + [None])

    kind = "scattermap" if _uses_scattermap() else "scattermapbox"
    fig.add_trace(S(type=kind, mode="lines", lon=lons, lat=lats,
                    line=dict(color=ROAD_EDGE, width=7),
                    hoverinfo="skip", showlegend=False))
    fig.add_trace(S(type=kind, mode="lines", lon=lons, lat=lats,
                    line=dict(color=ROAD_COLOR, width=5),
                    hoverinfo="skip", showlegend=False, name="Roads"))


def _add_blocked_roads(fig, env):
    """Render blocked roads as red hazard crosses."""
    S = _scatter_cls()
    if not env.blocked_cells:
        return
    lats, lons, texts = [], [], []
    for r, c in env.blocked_cells:
        lat, lon = to_latlon(r, c)
        lats.append(lat)
        lons.append(lon)
        texts.append(f"<b>Road blocked</b><br>({r},{c})")
    fig.add_trace(S(
        type=_kind(),
        mode="markers+text",
        lon=lons, lat=lats,
        marker=dict(size=16, color="#5f6368", symbol="x"),
        text=["\U0001F6D1"] * len(lats),
        textposition="middle center",
        textfont=dict(size=14, color="#d93025"),
        hovertext=texts, hoverinfo="text",
        name="Road Blocked"
    ))


def _add_risk_zones(fig, env):
    """Render risk zones as translucent hazard circles."""
    S = _scatter_cls()
    if not env.risk_zones:
        return
    lats, lons, texts, sizes = [], [], [], []
    for z in env.risk_zones:
        r, c = z['position']
        lat, lon = to_latlon(r, c)
        lats.append(lat)
        lons.append(lon)
        sizes.append(20 + z['risk_level'] * 26)
        texts.append(f"<b>Risk zone</b> ({r},{c})<br>"
                     f"Risk level: {z['risk_level']:.2f}")
    fig.add_trace(S(
        type=_kind(),
        mode="markers",
        lon=lons, lat=lats,
        marker=dict(size=sizes, color="rgba(251,146,60,0.45)"),
        hovertext=texts, hoverinfo="text",
        name="Risk Zone"
    ))


def _add_hospitals(fig, env):
    """Render hospitals as green map pins."""
    for i, (r, c) in enumerate(env.hospitals):
        lat, lon = to_latlon(r, c)
        _add_pin(fig, lat, lon, "#188038", "circle", "\U0001F3E5", 26,
                 f"<b>Hospital {i}</b><br>({r},{c})",
                 f"Hospital {i}", showlegend=(i == 0), textfont_size=13)


def _add_victims(fig, env):
    """Render waiting victims as red severity pins."""
    for v in env.victims:
        if v.get('rescued'):
            continue
        r, c = v['position']
        lat, lon = to_latlon(r, c)
        _add_pin(fig, lat, lon, "#d93025", "circle", str(v['severity']),
                 20 + v['severity'] * 2.2,
                 f"<b>Victim {v['id']}</b><br>Severity: {v['severity']}/10<br>"
                 f"Status: awaiting rescue",
                 "Victim", showlegend=False, textfont_size=10)


def _add_route(fig, path):
    """Draw the Google Maps style route: white casing under a blue line."""
    S = _scatter_cls()
    lats = [to_latlon(r, c)[0] for r, c in path]
    lons = [to_latlon(r, c)[1] for r, c in path]

    fig.add_trace(S(type=_kind(), mode="lines", lon=lons, lat=lats,
                    line=dict(color=ROUTE_CASING, width=11),
                    hoverinfo="skip", showlegend=False))
    fig.add_trace(S(type=_kind(), mode="lines", lon=lons, lat=lats,
                    line=dict(color=GOOGLE_BLUE, width=6),
                    hoverinfo="skip", name="Rescue Route"))

    # Direction chevrons along the route
    step = max(len(path) // 6, 1)
    for i in range(step, len(path) - 1, step):
        fig.add_trace(S(
            type=_kind(), mode="markers",
            lon=[lons[i]], lat=[lats[i]],
            marker=dict(size=9, color="white", symbol="triangle-up"),
            hoverinfo="skip", showlegend=False
        ))

    # Start + end markers
    _add_pin(fig, lats[0], lons[0], "#188038", "circle", "A", 20,
             "Start", "Start", showlegend=True, textfont_size=11)
    _add_pin(fig, lats[-1], lons[-1], "#d93025", "circle", "B", 20,
             "Destination", "Destination", showlegend=True, textfont_size=11)


def _finalize(fig, env, title, style, zoom=15):
    """Apply shared map layout with proper bounds and padding."""
    top_lat, _ = to_latlon(-1, 0)
    _, right_lon = to_latlon(0, env.cols)
    bottom_lat, _ = to_latlon(env.rows, 0)
    _, left_lon = to_latlon(0, -1)
    center_lat = (top_lat + bottom_lat) / 2
    center_lon = (left_lon + right_lon) / 2

    if _uses_scattermap():
        fig.update_layout(
            map=dict(
                style=style,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom
            )
        )
    else:
        fig.update_layout(
            mapbox=dict(
                style=style,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,
                attribution="© OpenStreetMap © CARTO"
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=17)),
        margin=dict(r=8, t=48, l=8, b=8),
        height=620,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#dadce0", borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12,
                        font_family="Roboto, Arial, sans-serif"),
        paper_bgcolor="white",
        dragmode="zoom"
    )


def render_map(env, path=None, title="Rescue Operation Map", style="carto-positron",
               zoom=15):
    """Render the full realistic map with every layer."""
    if env is None:
        return
    fig = go.Figure()
    path_cells = set(path) if path else set()

    _add_buildings(fig, env, path_cells)
    _add_road_network(fig, env, path_cells)
    if path:
        _add_route(fig, path)
    _add_risk_zones(fig, env)
    _add_hospitals(fig, env)
    _add_victims(fig, env)
    _add_blocked_roads(fig, env)

    _finalize(fig, env, title, style, zoom)
    st.plotly_chart(fig, width='stretch',
                    config={"displayModeBar": True,
                            "scrollZoom": True,
                            "showLegend": True})


# ============================================================================
# ANIMATED TRACKING (Google navigation style)
# ============================================================================
def render_tracking(env, path, title="Live Rescue Tracking",
                    style="carto-positron"):
    """Animate the rescue vehicle along the route like turn-by-turn nav."""
    if not path or len(path) < 2:
        st.info("No route to animate.")
        return

    S = _scatter_cls()

    lats = [to_latlon(r, c)[0] for r, c in path]
    lons = [to_latlon(r, c)[1] for r, c in path]

    # Rotating vehicle marker that points along the direction of travel
    tri = {"North": "triangle-up", "South": "triangle-down",
           "East": "triangle-right", "West": "triangle-left"}

    fig = go.Figure()
    _add_buildings(fig, env, set(path))
    _add_road_network(fig, env, set(path))
    _add_risk_zones(fig, env)
    _add_hospitals(fig, env)
    _add_victims(fig, env)
    _add_blocked_roads(fig, env)

    # Full route (dimmed) then the driven portion
    fig.add_trace(S(type=_kind(), mode="lines", lon=lons, lat=lats,
                    line=dict(color="#9aa0a6", width=6),
                    hoverinfo="skip", showlegend=False))
    fig.add_trace(S(type=_kind(), mode="lines",
                    lon=lons[:2], lat=lats[:2],
                    line=dict(color=ROUTE_CASING, width=11),
                    hoverinfo="skip", showlegend=False))
    fig.add_trace(S(type=_kind(), mode="lines",
                    lon=lons[:2], lat=lats[:2],
                    line=dict(color=GOOGLE_BLUE, width=6),
                    hoverinfo="skip", name="Route travelled"))

    _add_pin(fig, lats[0], lons[0], GOOGLE_BLUE, "circle", "\U0001F691", 24,
             "Rescue unit", "Rescue Unit", showlegend=True, textfont_size=13)
    _add_pin(fig, lats[-1], lons[-1], "#d93025", "circle", "", 19,
             "Victim", "Victim", showlegend=True)

    # Build animation frames
    frames = []
    for i in range(2, len(path) + 1):
        h = bearing_between(path[i - 2], path[i - 1])
        frames.append(go.Frame(
            data=[
                S(type=_kind(), mode="markers",
                  lon=[lons[i - 1]], lat=[lats[i - 1]],
                  marker=dict(size=30, color="white", symbol="circle",
                              opacity=0.95),
                  hoverinfo="skip", showlegend=False),
                S(type=_kind(), mode="lines", lon=lons[:i], lat=lats[:i],
                  line=dict(color=ROUTE_CASING, width=11),
                  hoverinfo="skip", showlegend=False),
                S(type=_kind(), mode="lines", lon=lons[:i], lat=lats[:i],
                  line=dict(color=GOOGLE_BLUE, width=6),
                  hoverinfo="skip", showlegend=False),
                S(type=_kind(), mode="markers", lon=[lons[i - 1]],
                  lat=[lats[i - 1]],
                  marker=dict(size=24, color=GOOGLE_BLUE, symbol=tri[h]),
                  text=["\U0001F691"], textposition="middle center",
                  textfont=dict(size=13), hoverinfo="skip", showlegend=False),
            ],
            name=str(i)
        ))

    fig.frames = frames

    m = path_metrics(path)
    _finalize(fig, env, title, style, zoom=15.4)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.02, y=1.10, xanchor="left", yanchor="top",
            pad=dict(t=2, b=2),
            buttons=[
                dict(label="\u25B6  Start route", method="animate",
                     args=[None, dict(frame=dict(duration=420, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=0))]),
                dict(label="\u23F8  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate",
                                       transition=dict(duration=0))]),
                dict(label="\u21BA  Reset", method="animate",
                     args=[[0], dict(frame=dict(duration=0, redraw=True),
                                     mode="immediate",
                                     transition=dict(duration=0))]),
            ]
        )],
        sliders=[dict(
            active=0, x=0.12, y=1.10, len=0.82, pad=dict(t=2, b=2),
            currentvalue=dict(prefix="Step: ", visible=True,
                              xanchor="right"),
            steps=[dict(method="animate",
                        label=str(i),
                        args=[[i], dict(frame=dict(duration=0, redraw=True),
                                        mode="immediate",
                                        transition=dict(duration=0))])
                   for i in range(len(path))]
        )]
    )

    st.plotly_chart(fig, width='stretch',
                    config={"displayModeBar": False})

    # Navigation info card - Google Maps bottom sheet style
    st.markdown(
        f"""
<div style="border:1px solid #dadce0;border-radius:14px;padding:16px 20px;
            font-family:Roboto,Arial,sans-serif;background:#fff;
            box-shadow:0 1px 4px rgba(0,0,0,.08)">
  <div style="font-size:17px;font-weight:600;color:#202124;margin-bottom:10px">
    Route overview
  </div>
  <div style="display:flex;gap:34px;flex-wrap:wrap">
    <div>
      <div style="font-size:11px;color:#5f6368;letter-spacing:.6px">DISTANCE</div>
      <div style="font-size:21px;font-weight:600;color:#202124">
        {m['meters']:.0f} m</div>
    </div>
    <div>
      <div style="font-size:11px;color:#5f6368;letter-spacing:.6px">DURATION</div>
      <div style="font-size:21px;font-weight:600;color:#202124">{m['duration']}</div>
    </div>
    <div>
      <div style="font-size:11px;color:#5f6368;letter-spacing:.6px">ETA</div>
      <div style="font-size:21px;font-weight:600;color:#202124">{m['eta']}</div>
    </div>
    <div>
      <div style="font-size:11px;color:#5f6368;letter-spacing:.6px">STYLE</div>
      <div style="font-size:21px;font-weight:600;color:#188038">Fastest</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    steps = get_turn_by_turn(path)
    if steps:
        st.markdown("**Turn-by-turn directions**")
        rows = "".join(
            f"<tr><td style='padding:6px 12px 6px 0;font-size:16px'>{s['icon']}</td>"
            f"<td style='padding:6px 0;font-size:14px;color:#202124'>{s['text']}</td></tr>"
            for s in steps)
        st.markdown(
            f"<div style='border:1px solid #dadce0;border-radius:12px;"
            f"padding:10px 16px;background:#fff'>\n<table style='border-collapse:collapse'>"
            f"{rows}</table></div>",
            unsafe_allow_html=True)


# ============================================================================
# FALLBACK RENDERER (works with no internet / no map tiles)
# ============================================================================
def render_offline_map(env, path=None, title="Rescue Map (offline mode)"):
    if env is None:
        return
    fig = go.Figure()

    for i in range(env.rows + 1):
        fig.add_trace(go.Scatter(x=[-0.5, env.cols - 0.5], y=[i - 0.5, i - 0.5],
                                 mode="lines", line=dict(color="#c8c6c1", width=1),
                                 hoverinfo="skip", showlegend=False))
    for j in range(env.cols + 1):
        fig.add_trace(go.Scatter(x=[j - 0.5, j - 0.5], y=[-0.5, env.rows - 0.5],
                                 mode="lines", line=dict(color="#c8c6c1", width=1),
                                 hoverinfo="skip", showlegend=False))

    for z in env.risk_zones:
        r, c = z['position']
        fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers",
                                 marker=dict(size=26 + z['risk_level'] * 24,
                                             color="rgba(251,146,60,0.5)"),
                                 name="Risk Zone", hoverinfo="skip"))
    for i, (r, c) in enumerate(env.hospitals):
        fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers",
                                 marker=dict(size=30, color="white",
                                             symbol="circle"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers+text",
                                 marker=dict(size=24, color="#188038",
                                             symbol="circle"),
                                 text=["\U0001F3E5"], textposition="middle center",
                                 name=f"Hospital {i}", hoverinfo="skip",
                                 showlegend=(i == 0)))
    for v in env.victims:
        if v.get('rescued'):
            continue
        r, c = v['position']
        size = 18 + v['severity'] * 2
        fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers",
                                 marker=dict(size=size + 6, color="white",
                                             symbol="circle"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers+text",
                                 marker=dict(size=size, color="#d93025",
                                             symbol="circle"),
                                 text=[str(v['severity'])], textposition="middle center",
                                 textfont=dict(color="white", size=10),
                                 name="Victim", hoverinfo="skip", showlegend=False))
    if env.blocked_cells:
        br, bc = zip(*env.blocked_cells)
        fig.add_trace(go.Scatter(x=list(bc), y=list(br), mode="markers",
                                 marker=dict(size=16, color="#5f6368", symbol="x"),
                                 name="Road Blocked", hoverinfo="skip"))
    if path and len(path) > 1:
        pr, pc = zip(*path)
        fig.add_trace(go.Scatter(x=list(pc), y=list(pr), mode="lines",
                                 line=dict(color="#ffffff", width=9), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=list(pc), y=list(pr), mode="lines",
                                 line=dict(color=GOOGLE_BLUE, width=5),
                                 name="Rescue Route", hoverinfo="skip"))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(range=[-1, env.cols], dtick=1, gridcolor="#f1f3f4",
                   zeroline=False, title="Column"),
        yaxis=dict(range=[-1, env.rows], dtick=1, gridcolor="#f1f3f4",
                   zeroline=False, title="Row", scaleanchor="x", scaleratio=1),
        height=600, plot_bgcolor="#f8f9fa", showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.9)", borderwidth=1)
    )
    st.plotly_chart(fig, width='stretch')


# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    defaults = {
        'environment': None, 'agent': None, 'ml_model': None,
        'rescue_log': [], 'mission_complete': False,
        'current_path': None, 'path_label': None, 'offline_mode': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.set_page_config(page_title="AI Disaster Rescue System",
                       page_icon="\U0001F691", layout="wide")
    init_session_state()

    # ---- Google-style header ----
    st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:2px">
  <div style="font-size:38px">\U0001F691</div>
  <div>
    <div style="font-size:26px;font-weight:700;color:#202124;
                font-family:Roboto,Arial,sans-serif">
      AI Disaster Rescue System</div>
    <div style="font-size:13.5px;color:#5f6368;font-family:Roboto,Arial,sans-serif">
      Search &middot; CSP &middot; Machine Learning &middot; Fuzzy Logic &middot;
      Dynamic Replanning</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.divider()

    # ================= SIDEBAR =================
    st.sidebar.header("\U0001F527 Simulation setup")

    rows = st.sidebar.slider("Grid rows", 5, 20, 10)
    cols = st.sidebar.slider("Grid columns", 5, 20, 10)
    n_victims = st.sidebar.slider("Victims", 1, 10, 5)
    n_hospitals = st.sidebar.slider("Hospitals", 1, 4, 2)
    n_risk = st.sidebar.slider("Risk zones", 0, 8, 3)
    n_blocked = st.sidebar.slider("Blocked roads", 0, 15, 8)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

    st.sidebar.divider()
    st.sidebar.subheader("\U0001F5FA Map view")
    style_label = st.sidebar.radio("Base map", list(MAP_STYLES.keys()),
                                   index=0)
    style = MAP_STYLES[style_label]
    st.sidebar.toggle("Offline mode", key="offline_mode",
                      help="Render without map tiles (no internet needed)")

    if st.sidebar.button("\U0001F504 Generate environment",
                         type="primary", width='stretch'):
        env = Environment(rows, cols, n_victims, n_hospitals,
                          n_risk, n_blocked, seed)
        st.session_state.environment = env
        st.session_state.agent = RescueAgent(env, env.hospitals)
        st.session_state.rescue_log = []
        st.session_state.mission_complete = False
        st.session_state.current_path = None
        st.session_state.path_label = None
        st.rerun()

    # ================= TABS =================
    tabs = st.tabs(["\U0001F5FA Live map", "\U0001F50D Pathfinding",
                    "\U0001F691 CSP allocation", "\U0001F916 ML models",
                    "\U0001F9E0 Fuzzy logic", "\U0001F3AE Simulation"])

    def draw_map(env, path, title, zoom=15):
        if st.session_state.offline_mode:
            render_offline_map(env, path, title + " (offline)")
        else:
            try:
                render_map(env, path, title, style, zoom)
            except Exception as e:  # pragma: no cover
                st.warning(f"Map tiles unavailable ({e.__class__.__name__}). "
                           "Switched to offline renderer.")
                render_offline_map(env, path, title + " (offline)")

    # ---------------- TAB 1: LIVE MAP ----------------
    with tabs[0]:
        if env := st.session_state.environment:
            st.subheader("Operational map")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Victims waiting", len(env.get_unrescued_victims()))
            c2.metric("Hospitals", len(env.hospitals))
            c3.metric("Risk zones", len(env.risk_zones))
            c4.metric("Blocked roads", len(env.blocked_cells))

            path = st.session_state.current_path
            title = st.session_state.path_label or "Disaster area overview"
            draw_map(env, path, title)

            st.caption("Legend: \U0001F3E5 hospital  \U0001F534 victim "
                       "(number = severity)  \U0001F7E1 risk zone  "
                       "\U0001F6D1 blocked road  \U0001F697 rescue unit")

            if path:
                m = path_metrics(path)
                m1, m2, m3 = st.columns(3)
                m1.metric("Route distance", f"{m['meters']:.0f} m")
                m2.metric("Estimated time", m['duration'])
                m3.metric("Path steps", m['steps'])

            st.divider()
            st.subheader("Victim registry")
            rows_data = [{
                "ID": v['id'],
                "Position": f"({v['position'][0]}, {v['position'][1]})",
                "Severity": v['severity'],
                "Risk zone": f"{env.get_risk_at(*v['position']):.2f}",
                "Status": "Rescued" if v.get('rescued') else "Waiting",
            } for v in env.victims]
            st.dataframe(pd.DataFrame(rows_data), width='stretch',
                         hide_index=True)
        else:
            st.info("Press **Generate environment** in the sidebar to begin.")

    # ---------------- TAB 2: PATHFINDING ----------------
    with tabs[1]:
        if env := st.session_state.environment:
            st.subheader("Pathfinding")

            algo = st.selectbox("Algorithm", [
                "A* (recommended)", "Risk-Aware A*", "BFS", "DFS",
                "Greedy Best-First", "Hill Climbing"])

            a, b, c, d = st.columns(4)
            sr = a.number_input("Start row", 0, env.rows - 1, 0)
            sc = b.number_input("Start col", 0, env.cols - 1, 0)
            gr = c.number_input("Goal row", 0, env.rows - 1, env.rows - 1)
            gc = d.number_input("Goal col", 0, env.cols - 1, env.cols - 1)

            start, goal = (int(sr), int(sc)), (int(gr), int(gc))

            if st.button("\U0001F50D Find route", type="primary"):
                fn = {"A* (recommended)": astar, "Risk-Aware A*": risk_aware_astar,
                      "BFS": bfs, "DFS": dfs, "Greedy Best-First": greedy_best_first,
                      "Hill Climbing": hill_climbing}[algo]
                path, cost = fn(env, start, goal)
                if path:
                    st.session_state.current_path = path
                    st.session_state.path_label = f"{algo} route \u00b7 {len(path)-1} steps \u00b7 cost {cost}"
                    st.success(f"Route found in {len(path)-1} steps (cost {cost}).")
                else:
                    st.session_state.current_path = None
                    st.error("No route exists to that destination.")

            if st.session_state.current_path and st.session_state.path_label:
                draw_map(env, st.session_state.current_path,
                         st.session_state.path_label)
                render_tracking(env, st.session_state.current_path,
                                f"Tracking \u00b7 {st.session_state.path_label}")

            st.divider()
            if st.button("\U0001F4CA Compare all algorithms"):
                data = []
                for name, fn in [("BFS", bfs), ("DFS", dfs), ("A*", astar),
                                 ("Risk-Aware A*", risk_aware_astar),
                                 ("Greedy Best-First", greedy_best_first),
                                 ("Hill Climbing", hill_climbing)]:
                    p, co = fn(env, start, goal)
                    data.append({"Algorithm": name, "Cost": co,
                                 "Steps": len(p) - 1 if p else None,
                                 "Reachable": "\u2713" if p else "\u2717"})
                st.dataframe(pd.DataFrame(data), width='stretch',
                             hide_index=True)
        else:
            st.info("Generate an environment first.")

    # ---------------- TAB 3: CSP ----------------
    with tabs[2]:
        if env := st.session_state.environment:
            st.subheader("CSP ambulance allocation")
            capacity = st.slider("Ambulance capacity", 1, 5, 3)

            b1, b2 = st.columns(2)
            if b1.button("\U0001F691 Solve (backtracking + MRV)", type="primary",
                         width='stretch'):
                alloc = CSPAmbulanceAllocator(env.victims, env.hospitals, capacity)
                assignment = alloc.backtracking_search(use_mrv=True)
                if assignment:
                    st.session_state.csp_assignment = assignment
                    st.session_state.csp_details = alloc.get_assignment_details(assignment)
                    st.success(f"Solution found \u00b7 total distance "
                               f"{alloc.get_total_cost(assignment)}")
                else:
                    st.error("No feasible assignment (capacity too low).")

            if b2.button("\U0001F504 Solve without MRV", width='stretch'):
                alloc = CSPAmbulanceAllocator(env.victims, env.hospitals, capacity)
                assignment = alloc.backtracking_search(use_mrv=False)
                if assignment:
                    st.session_state.csp_assignment = assignment
                    st.session_state.csp_details = alloc.get_assignment_details(assignment)
                    st.success(f"Solution found \u00b7 total distance "
                               f"{alloc.get_total_cost(assignment)}")
                else:
                    st.error("No feasible assignment (capacity too low).")

            if 'csp_details' in st.session_state and st.session_state.csp_details:
                st.dataframe(pd.DataFrame([
                    {"Victim": d['victim_id'],
                     "Victim pos": f"{d['victim_position']}",
                     "Severity": d['severity'],
                     "Ambulance": d['ambulance_id'],
                     "Hospital pos": f"{d['hospital_position']}",
                     "Distance": d['distance']}
                    for d in st.session_state.csp_details
                ]), width='stretch', hide_index=True)

                d0 = st.session_state.csp_details[0]
                p, _ = astar(env, d0['hospital_position'], d0['victim_position'])
                if p:
                    draw_map(env, p, f"Ambulance {d0['ambulance_id']} \u2192 Victim {d0['victim_id']}")
        else:
            st.info("Generate an environment first.")

    # ---------------- TAB 4: ML ----------------
    with tabs[3]:
        st.subheader("Machine learning priority prediction")

        def ensure_ml():
            if st.session_state.ml_model is None:
                with st.spinner("Training models\u2026"):
                    ml = RescueMLModel(n_samples=500)
                    ml.generate_dataset()
                    ml.train_models()
                    st.session_state.ml_model = ml
            return st.session_state.ml_model

        p1, p2, p3, p4 = st.columns(4)
        sev = p1.slider("Severity", 1, 10, 7)
        dist = p2.slider("Distance", 1, 20, 6)
        risk = p3.slider("Risk level", 0.0, 1.0, 0.6, 0.05)
        blk = p4.slider("Blockage proximity", 0, 4, 1)

        if st.button("\U0001F916 Predict priority", type="primary"):
            ml = ensure_ml()
            priority, votes = ml.predict_priority(int(sev), int(dist), float(risk), int(blk))
            colour = {"HIGH": "#d93025", "MEDIUM": "#e8710a", "LOW": "#188038"}[priority]
            st.markdown(
                f"""<div style="border:1px solid #dadce0;border-left:6px solid {colour};
                border-radius:12px;padding:16px 22px;background:#fff;margin-top:8px">
                <div style="font-size:12px;color:#5f6368;letter-spacing:.8px">
                  MAJORITY VOTE RESULT</div>
                <div style="font-size:34px;font-weight:700;color:{colour};
                            font-family:Roboto,Arial,sans-serif">{priority}</div>
                <div style="font-size:13px;color:#5f6368">
                  kNN: {votes['kNN']} &nbsp;|&nbsp; Naive Bayes: {votes['Naive Bayes']}
                  &nbsp;|&nbsp; Decision Tree: {votes['Decision Tree']}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        if st.button("\U0001F4CA Evaluate models"):
            ml = ensure_ml()
            metrics = ml.evaluate_models()
            st.dataframe(pd.DataFrame([
                {"Model": k,
                 "Accuracy": f"{v['accuracy']:.4f}",
                 "Precision": f"{v['precision']:.4f}",
                 "Recall": f"{v['recall']:.4f}",
                 "F1": f"{v['f1_score']:.4f}"}
                for k, v in metrics.items()
            ]), width='stretch', hide_index=True)

            fig = make_cm_figure(metrics)
            st.plotly_chart(fig, width='stretch')

    # ---------------- TAB 5: FUZZY ----------------
    with tabs[4]:
        st.subheader("Fuzzy logic under uncertainty")

        f1, f2, f3, f4 = st.columns(4)
        f_dist = f1.slider("Distance", 0, 20, 5)
        f_risk = f2.slider("Risk", 0.0, 1.0, 0.5, 0.05)
        f_sev = f3.slider("Severity", 0, 10, 6)
        f_blk = f4.slider("Blockage probability", 0.0, 1.0, 0.3, 0.05)

        if st.button("\U0001F9E0 Evaluate", type="primary"):
            fz = FuzzyRescueDecision()
            res = fz.evaluate(f_dist, f_risk, f_sev, f_blk)
            risk_path = fz.should_use_risk_aware_path(f_risk, f_blk)

            a, b = st.columns([1, 1])
            a.metric("Priority score", res['priority_score'])
            b.metric("Category", res['category'])
            st.write("**Navigation mode:**", "Risk-Aware A*" if risk_path else "A* (fastest)")

            st.markdown("**Fuzzified inputs**")
            st.json(res['fuzzy_inputs'], expanded=False)

            st.markdown(f"**Fired rules ({len(res['rule_activations'])})**")
            if res['rule_activations']:
                st.dataframe(pd.DataFrame([
                    {"Rule": r['rule'], "Activation": round(r['activation'], 3)}
                    for r in res['rule_activations']
                ]), width='stretch', hide_index=True)
            else:
                st.caption("No rules fired for this input.")

            st.plotly_chart(membership_figure(fz, f_dist, f_risk, f_sev, f_blk, res),
                            width='stretch')

    # ---------------- TAB 6: SIMULATION ----------------
    with tabs[5]:
        if env := st.session_state.environment:
            st.subheader("Rescue mission")

            if st.button("\u25B6 Run full rescue mission", type="primary"):
                agent = RescueAgent(env, env.hospitals)
                log = agent.run_rescue_mission()
                st.session_state.agent = agent
                st.session_state.rescue_log = log
                st.session_state.mission_complete = True
                st.rerun()

            if st.session_state.mission_complete:
                stats = st.session_state.agent.get_statistics()
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Rescued", stats['rescued_victims'])
                s2.metric("Remaining", stats['remaining_victims'])
                s3.metric("Total distance", f"{stats['total_distance']*CELL_KM*1000:.0f} m")
                s4.metric("Missions", stats['missions_completed'])

                log_rows = [{
                    "Victim": e.get('victim_id'),
                    "Status": e.get('status'),
                    "Severity": next((v['severity'] for v in env.victims
                                      if v['id'] == e.get('victim_id')), None),
                    "Algorithm": e.get('algorithm_to_victim'),
                    "Distance": e.get('total_distance'),
                } for e in st.session_state.rescue_log]
                st.dataframe(pd.DataFrame(log_rows), width='stretch',
                             hide_index=True)

                first = next((e for e in st.session_state.rescue_log
                              if e.get('path_to_victim')), None)
                if first:
                    p = first['path_to_victim'] + first['path_to_hospital'][1:]
                    draw_map(env, p, f"Mission route \u00b7 Victim {first['victim_id']}")
                    render_tracking(env, p, f"Tracking mission to victim {first['victim_id']}")

            st.divider()
            st.subheader("Dynamic replanning scenarios")

            def free_cells():
                return [(r, c) for r in range(env.rows) for c in range(env.cols)
                        if env.grid[r][c] == env.EMPTY]

            d1, d2, d3, d4 = st.columns(4)
            if d1.button("\U0001F6A7 Block road", width='stretch'):
                cells = free_cells()
                if cells:
                    r, c = cells[len(cells) // 2]
                    env.add_blocked_road(r, c)
                    st.session_state.current_path = None
                    st.rerun()
            if d2.button("⚠️ Raise risk", width='stretch'):
                cells = free_cells()
                if cells:
                    r, c = cells[len(cells) // 3]
                    env.increase_risk(r, c, 0.3)
                    st.rerun()
            if d3.button("\U0001F195 New victim", width='stretch'):
                cells = free_cells()
                if cells:
                    r, c = cells[-2]
                    env.add_victim(r, c)
                    st.rerun()
            if d4.button("\U0001F50B Reallocate (CSP)", width='stretch'):
                alloc = CSPAmbulanceAllocator(env.victims, env.hospitals, 3)
                a = alloc.backtracking_search(use_mrv=True)
                if a:
                    st.success(f"Reassigned {len(a)} victims \u00b7 cost "
                               f"{alloc.get_total_cost(a)}")
                else:
                    st.error("Insufficient ambulance capacity.")
        else:
            st.info("Generate an environment first.")


def make_cm_figure(metrics):
    """Confusion matrices side by side."""
    names = list(metrics.keys())
    fig = make_subplots(rows=1, cols=len(names),
                        subplot_titles=[n.replace(" (Majority Vote)", "\nEnsemble")
                                        for n in names])
    labels = ["LOW", "MEDIUM", "HIGH"]
    for i, n in enumerate(names, 1):
        cm = metrics[n]['confusion_matrix']
        fig.add_trace(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues",
                                 showscale=(i == len(names)),
                                 text=cm, texttemplate="%{text}",
                                 hovertemplate="true %{y}<br>pred %{x}: %{z}<extra></extra>"),
                      row=1, col=i)
    fig.update_layout(height=340, margin=dict(t=60, b=40, l=40, r=20),
                      title="Confusion matrices")
    return fig


def membership_figure(fz, dist, risk, sev, blk, res):
    """Plot the four membership functions and the aggregated output."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Distance", "Risk", "Severity", "Blockage probability"])

    series = [
        ([fz.distance_near, fz.distance_medium, fz.distance_far],
         ["near", "medium", "far"], dist, np.linspace(0, 20, 200)),
        ([fz.risk_low, fz.risk_medium, fz.risk_high],
         ["low", "medium", "high"], risk, np.linspace(0, 1, 200)),
        ([fz.severity_low, fz.severity_medium, fz.severity_high],
         ["low", "medium", "high"], sev, np.linspace(0, 10, 200)),
        ([fz.blockage_low, fz.blockage_high],
         ["low", "high"], blk, np.linspace(0, 1, 200)),
    ]
    pos = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (fns, labels, value, xs), (row, col) in zip(series, pos):
        for fn, lab in zip(fns, labels):
            y = [float(v) for v in fn(xs)]
            fig.add_trace(go.Scatter(y=y, x=xs, mode="lines", name=lab,
                                     showlegend=False), row=row, col=col)
        fig.add_vline(x=value, line_dash="dash", line_color="#d93025",
                      row=row, col=col)
        fig.update_xaxes(title_text="input", row=row, col=col)
        fig.update_yaxes(title_text="\u03bc", range=[0, 1.05], row=row, col=col)

    fig.update_layout(height=460, margin=dict(t=50, b=40, l=50, r=20),
                      title="Fuzzification (red dashed line = crisp input)")
    return fig


if __name__ == "__main__":
    main()
