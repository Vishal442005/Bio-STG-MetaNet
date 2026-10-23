"""
brain_topo_viewer.py  (v2 - Premium Visual Edition)
====================================================
3D Brain EEG Topography Viewer.

NEW in v2:
  - Colored anatomical lobe sections (Frontal/Parietal/Temporal/Occipital/Central)
  - Translucent frosted-glass outer skull shell
  - Glowing electrode markers with tooltip labels
  - Animated EEG activity overlay (Play / Scrub / Loop)
  - Premium dark-mode HTML with custom UI panel

Run from Internship folder:
    python brain_topo_viewer.py
"""

import os, gzip, random, sys, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
from scipy.interpolate import RBFInterpolator

warnings.filterwarnings('ignore')

# ==================================================================
# ELECTRODE POSITIONS  (standard 10-20 system on a unit sphere)
#   theta = angle from vertex  (0 = CZ top, 90 = ear level)
#   phi   = azimuth from front (0 = Nasion/front, 90 = right,
#                               180 = back,  270 = left)
# ==================================================================
ELEC_POS = {
    'FP1':(18,342), 'FP2':(18,18),  'FPZ':(18,0),
    'AF7':(36,308), 'AF8':(36,52),
    'AF1':(36,345), 'AF2':(36,15),  'AFZ':(36,0),
    'F7' :(54,308), 'F8' :(54,52),
    'F5' :(54,326), 'F6' :(54,34),
    'F3' :(54,342), 'F4' :(54,18),
    'F1' :(54,352), 'F2' :(54,8),   'FZ' :(54,0),
    'FT7':(72,308), 'FT8':(72,52),
    'FC5':(72,328), 'FC6':(72,32),
    'FC3':(72,344), 'FC4':(72,16),
    'FC1':(72,352), 'FC2':(72,8),   'FCZ':(72,0),
    'T7' :(90,270), 'T8' :(90,90),
    'T9' :(90,252), 'T10':(90,108),
    'C5' :(90,288), 'C6' :(90,72),
    'C3' :(90,306), 'C4' :(90,54),
    'C1' :(90,324), 'C2' :(90,36),
    'CZ' :(90,0),
    'TP7':(108,270),'TP8':(108,90),
    'CP5':(108,288),'CP6':(108,72),
    'CP3':(108,306),'CP4':(108,54),
    'CP1':(108,324),'CP2':(108,36), 'CPZ':(108,0),
    'P9' :(126,252),'P10':(126,108),
    'P7' :(126,270),'P8' :(126,90),
    'P5' :(126,288),'P6' :(126,72),
    'P3' :(126,306),'P4' :(126,54),
    'P1' :(126,324),'P2' :(126,36), 'PZ' :(126,0),
    'PO7':(144,270),'PO8':(144,90),
    'PO1':(144,324),'PO2':(144,36), 'POZ':(144,0),
    'O1' :(162,306),'O2' :(162,54),
    'O9' :(162,270),'O10':(162,90), 'OZ' :(162,0),
}

EXCLUDED = {'EKG','X','VEO','HEO','EMG','CB1','CB2','REF','GND'}

# ==================================================================
# LOBE COLOUR MAP
#   Discrete colorscale: 5 lobes mapped to IDs 0-4
# ==================================================================
LOBE_NAMES  = ['Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital']
LOBE_HEX    = ['#3b82f6', '#a855f7', '#10b981',  '#f59e0b',  '#ef4444']

# Discrete 5-step colorscale — all values strictly within [0, 1]
LOBE_COLORSCALE = [
    [0.00, '#3b82f6'],  # Frontal   (value 0)
    [0.24, '#3b82f6'],
    [0.25, '#a855f7'],  # Central   (value 1)
    [0.49, '#a855f7'],
    [0.50, '#10b981'],  # Parietal  (value 2)
    [0.74, '#10b981'],
    [0.75, '#f59e0b'],  # Temporal  (value 3)
    [0.99, '#f59e0b'],
    [1.00, '#ef4444'],  # Occipital (value 4)
]



# ==================================================================
# GEOMETRY HELPERS
# ==================================================================
def sph_to_xyz(theta_deg, phi_deg, r=1.0):
    """
    theta : angle from vertex (0=CZ top, 90=equator)
    phi   : azimuth from Nasion (0=front, 90=right ear)
    Axes: X=right, Y=up (CZ direction), Z=front (Nasion direction)
    """
    t = np.radians(theta_deg)
    p = np.radians(phi_deg)
    return np.array([
        r * np.sin(t) * np.sin(p),   # X right
        r * np.cos(t),               # Y up
        r * np.sin(t) * np.cos(p),   # Z front
    ])


def make_scalp_mesh(r=1.0, nt=70, np_=110, max_deg=118):
    """
    Slightly ellipsoidal skull cap.
    Returns (X, Y, Z) arrays shaped (nt, np_).
    Coordinate: X=right, Y=up, Z=front
    """
    theta = np.linspace(1e-6, np.radians(max_deg), nt)
    phi   = np.linspace(0, 2 * np.pi, np_)
    T, P  = np.meshgrid(theta, phi, indexing='ij')

    # Slight ellipse: head is taller than wide
    rx, ry, rz = r * 0.92, r, r * 0.88

    X = rx * np.sin(T) * np.sin(P)
    Y = ry * np.cos(T)
    Z = rz * np.sin(T) * np.cos(P)
    return X, Y, Z, T, P


def get_lobe_matrix(T, P):
    """
    Assign lobe ID (0-4) to each (theta, phi) grid point.
    theta, phi both in radians.
    """
    t_deg = np.degrees(T)
    p_deg = np.degrees(P) % 360

    lobe = np.zeros(T.shape, dtype=float)   # default = Frontal (0)

    # --- Temporal: sides of head ---
    temporal = (
        ((p_deg > 48) & (p_deg < 132) |         # right side
         (p_deg > 228) & (p_deg < 312)) &        # left side
        (t_deg > 58) & (t_deg < 116)
    )

    # --- Occipital: back-bottom ---
    occipital = (
        (p_deg > 125) & (p_deg < 235) &
        (t_deg > 105)
    )

    # --- Parietal: back-top (above occipital, behind central) ---
    parietal = (
        (p_deg > 110) & (p_deg < 250) &
        (t_deg > 68) & (t_deg <= 115) &
        ~temporal
    )

    # --- Central / Motor strip: midline band ---
    central = (
        ((p_deg < 38) | (p_deg > 322) |
         ((p_deg > 142) & (p_deg < 218))) &
        (t_deg > 52) & (t_deg < 90)
    )

    lobe[temporal]  = 3   # amber
    lobe[occipital] = 4   # red
    lobe[parietal]  = 2   # green
    lobe[central]   = 1   # purple
    # frontal (0) is the default

    return lobe


# ==================================================================
# FILE SELECTION
# ==================================================================
def pick_file(all_files, n=4):
    choices = random.sample(all_files, min(n, len(all_files)))

    def lbl(f):
        fn = os.path.basename(f)
        if fn.startswith('co2a'): return 'ALCOHOLIC'
        if fn.startswith('co2c'): return 'CONTROL  '
        return 'UNKNOWN  '

    print('\n' + '=' * 64)
    print('   BRAIN EEG TOPOGRAPHY VIEWER  --  Select a Trial')
    print('=' * 64)
    for i, f in enumerate(choices, 1):
        fn  = os.path.basename(f)
        sub = os.path.basename(os.path.dirname(f))
        print(f'\n  [{i}]  {fn}')
        print(f'       Subject : {sub}')
        print(f'       Class   : {lbl(f)}')
    print('\n' + '-' * 64)

    while True:
        try:
            idx = int(input(f'  Enter choice (1-{len(choices)}): ').strip()) - 1
            if 0 <= idx < len(choices):
                print(f'\n  >> Selected: {os.path.basename(choices[idx])}\n')
                return choices[idx]
            print(f'  Enter a number 1-{len(choices)}.')
        except ValueError:
            print('  Invalid.')


# ==================================================================
# EEG FILE READER (SMNI_CMI_TRAIN .gz format)
# ==================================================================
def read_eeg(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        lines = [l.strip().split() for l in f
                 if not l.startswith('#') and l.strip()]
    df = pd.DataFrame(lines, columns=['trial', 'channel', 't', 'v'])
    df['v'] = df['v'].astype(float)
    df['t'] = df['t'].astype(int)
    return df.pivot(index='channel', columns='t', values='v')


# ==================================================================
# BUILD FIGURE
# ==================================================================
def build_figure(filepath, n_anim_frames=40):
    fname     = os.path.basename(filepath)
    subject   = os.path.basename(os.path.dirname(filepath))
    alcoholic = fname.startswith('co2a')
    class_lbl = 'ALCOHOLIC' if alcoholic else 'CONTROL'

    # ── Load & filter ──────────────────────────────────────────
    print('Loading EEG data ...')
    matrix = read_eeg(filepath)   # channels x 256 time steps

    fs = 256
    b, a = butter(4, [0.5 / (fs/2), 100.0 / (fs/2)], btype='band')
    for ch in matrix.index:
        matrix.loc[ch] = filtfilt(b, a, matrix.loc[ch].values)

    n_time = matrix.shape[1]

    # ── Map electrodes to sphere ────────────────────────────────
    valid_ch, elec_xyz = [], []
    for ch in matrix.index:
        cu = ch.upper().strip()
        if cu in EXCLUDED or cu not in ELEC_POS:
            continue
        valid_ch.append(ch)
        elec_xyz.append(sph_to_xyz(*ELEC_POS[cu]))

    elec_xyz = np.array(elec_xyz)   # (n_elec, 3)
    print(f'  {len(valid_ch)} electrodes mapped to head positions.')

    # ── Build scalp mesh ────────────────────────────────────────
    mX, mY, mZ, T_mesh, P_mesh = make_scalp_mesh(r=1.0, nt=70, np_=110)
    mesh_pts = np.column_stack([mX.ravel(), mY.ravel(), mZ.ravel()])

    # ── Compute lobe colour matrix (static) ────────────────────
    lobe_matrix = get_lobe_matrix(T_mesh, P_mesh)

    # ── RBF interpolation for each animation frame ──────────────
    time_indices = np.linspace(0, n_time - 1, n_anim_frames).astype(int)
    print(f'Computing {n_anim_frames} animation frames (RBF interpolation) ...')

    frames_data = []
    for idx_i, ti in enumerate(time_indices):
        if (idx_i + 1) % max(1, n_anim_frames // 4) == 0:
            pct = int((idx_i + 1) / n_anim_frames * 100)
            print(f'     {pct:3d}% complete')

        vals = np.array([matrix.loc[ch, ti] for ch in valid_ch])
        rbf  = RBFInterpolator(elec_xyz, vals, kernel='thin_plate_spline', smoothing=1.0)
        interp = rbf(mesh_pts).reshape(mX.shape)
        frames_data.append(interp)

    # Normalise across all frames for consistent colorscale
    global_max = max(abs(fd).max() for fd in frames_data)
    if global_max < 1e-9:
        global_max = 1.0

    # ─────────────────────────────────────────────────────────────
    # BUILD TRACES
    # ─────────────────────────────────────────────────────────────

    # 1. LOBE BASE LAYER (static, fully opaque)
    lobe_trace = go.Surface(
        x=mX * 0.97, y=mY * 0.97, z=mZ * 0.97,
        surfacecolor=lobe_matrix,
        colorscale=LOBE_COLORSCALE,
        cmin=0, cmax=4,
        opacity=1.0,
        showscale=False,
        hoverinfo='skip',
        name='Lobes',
        lighting=dict(
            ambient=0.55, diffuse=0.7,
            specular=0.5, roughness=0.4,
            fresnel=0.2
        ),
        lightposition=dict(x=200, y=300, z=400),
    )

    # 2. EEG ACTIVITY LAYER (animated, semi-transparent)
    eeg_colorscale = [
        [0.0,  '#1a1aff'],  # strong negative -> deep blue
        [0.2,  '#4488ff'],
        [0.4,  '#aaddff'],  # near zero -> light blue
        [0.5,  '#ffffff'],  # zero -> white
        [0.6,  '#ffddaa'],
        [0.8,  '#ff5500'],
        [1.0,  '#cc0000'],  # strong positive -> deep red
    ]

    first_frame = frames_data[0]
    eeg_trace = go.Surface(
        x=mX, y=mY, z=mZ,
        surfacecolor=first_frame,
        colorscale=eeg_colorscale,
        cmin=-global_max, cmax=global_max,
        opacity=0.78,
        showscale=True,
        colorbar=dict(
            title=dict(text='uV', font=dict(color='#94a3b8', size=12)),
            tickfont=dict(color='#94a3b8', size=10),
            thickness=14,
            len=0.55, y=0.5,
            bgcolor='rgba(15,23,42,0.7)',
            bordercolor='#334155',
        ),
        name='EEG Activity',
        hoverinfo='skip',
        lighting=dict(
            ambient=0.6, diffuse=0.8,
            specular=0.3, roughness=0.5,
        ),
        lightposition=dict(x=200, y=300, z=400),
    )

    # 3. GLASS OUTER SHELL (translucent, slightly larger radius)
    gX, gY, gZ, _, _ = make_scalp_mesh(r=1.065, nt=45, np_=72)
    glass_trace = go.Surface(
        x=gX, y=gY, z=gZ,
        surfacecolor=np.ones_like(gX) * 0.5,
        colorscale=[[0, 'rgba(180,210,255,1)'], [1, 'rgba(180,210,255,1)']],
        opacity=0.06,
        showscale=False,
        hoverinfo='skip',
        name='Skull Shell',
        lighting=dict(
            ambient=0.2, diffuse=0.5,
            specular=2.0, roughness=0.05,
            fresnel=1.5
        ),
        lightposition=dict(x=-150, y=400, z=300),
    )

    # 4. ELECTRODE GLOW (outer larger dots, low opacity)
    ex, ey, ez = elec_xyz[:,0]*1.01, elec_xyz[:,1]*1.01, elec_xyz[:,2]*1.01
    elec_glow = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode='markers',
        marker=dict(size=7, color='#60a5fa', opacity=0.18,
                    line=dict(width=0)),
        hoverinfo='skip', showlegend=False,
    )

    # 5. ELECTRODES (solid dots with hover labels)
    elec_core = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode='markers+text',
        marker=dict(size=3.5, color='#e2e8f0', opacity=1.0,
                    line=dict(color='#60a5fa', width=1.5)),
        text=[ch.upper() for ch in valid_ch],
        textfont=dict(size=1, color='rgba(0,0,0,0)'),   # invisible text, shown via hover
        hovertext=[f'<b>{ch.upper()}</b><br>Pos: {ELEC_POS[ch.upper().strip()]}'
                   for ch in valid_ch],
        hoverinfo='text',
        showlegend=False,
        name='Electrodes',
    )

    # 6. NOSE
    nose_pts = [sph_to_xyz(12, d, r=1.05) for d in np.linspace(-18, 18, 8)]
    nose_tr = go.Scatter3d(
        x=[p[0] for p in nose_pts], y=[p[1] for p in nose_pts], z=[p[2] for p in nose_pts],
        mode='lines',
        line=dict(color='#94a3b8', width=3),
        hoverinfo='skip', showlegend=False,
    )

    # 7. EARS
    def ear_loop(phi_center_deg):
        offsets = [-8,-14,-11,-4,+4,+11,+8,-8]
        thetas  = [ 84, 87,  91, 93, 92,  90, 85, 84]
        pts = [sph_to_xyz(th, phi_center_deg + dp, r=1.09)
               for th, dp in zip(thetas, offsets)]
        return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

    ears = []
    for phi_c, side in [(90, 'Right Ear'), (270, 'Left Ear')]:
        ex2, ey2, ez2 = ear_loop(phi_c)
        ears.append(go.Scatter3d(x=ex2, y=ey2, z=ez2, mode='lines',
                                 line=dict(color='#64748b', width=3),
                                 hoverinfo='skip', showlegend=False))

    # 8. BRAIN LOBE LABELS (floating text above surface)
    lobe_label_pts = {
        'Frontal':   sph_to_xyz(38,  0,  r=1.22),
        'Central':   sph_to_xyz(75,  0,  r=1.22),
        'Parietal':  sph_to_xyz(95, 180, r=1.22),
        'Temporal L':sph_to_xyz(92, 270, r=1.22),
        'Temporal R':sph_to_xyz(92,  90, r=1.22),
        'Occipital': sph_to_xyz(145,180, r=1.22),
    }
    lobe_colors_label = {
        'Frontal':   '#3b82f6',
        'Central':   '#a855f7',
        'Parietal':  '#10b981',
        'Temporal L':'#f59e0b',
        'Temporal R':'#f59e0b',
        'Occipital': '#ef4444',
    }
    lx = [v[0] for v in lobe_label_pts.values()]
    ly = [v[1] for v in lobe_label_pts.values()]
    lz = [v[2] for v in lobe_label_pts.values()]
    lt = list(lobe_label_pts.keys())
    lc = [lobe_colors_label[k] for k in lt]

    lobe_labels_tr = go.Scatter3d(
        x=lx, y=ly, z=lz,
        mode='text',
        text=lt,
        textfont=dict(size=12, color=lc, family='Arial Black'),
        hoverinfo='skip', showlegend=False,
    )

    # 9. LEGEND items for lobes (dummy scatter3d)
    legend_traces = []
    for name, col in zip(LOBE_NAMES, LOBE_HEX):
        legend_traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=8, color=col),
            name=name, showlegend=True,
        ))

    # ── Assemble figure ─────────────────────────────────────────
    all_traces = (
        [lobe_trace, eeg_trace, glass_trace,
         elec_glow, elec_core,
         nose_tr] + ears + [lobe_labels_tr] + legend_traces
    )

    # EEG surface is trace index 1 → only it changes between frames
    eeg_idx = 1

    fig = go.Figure(data=all_traces)

    # ── Animation frames ────────────────────────────────────────
    frames = []
    for fi, fd in enumerate(frames_data):
        t_ms = int(time_indices[fi] / 256 * 1000)
        frames.append(go.Frame(
            data=[go.Surface(surfacecolor=fd)],
            traces=[eeg_idx],
            name=str(fi),
            layout=go.Layout(
                title_text=(f'<b>Brain EEG Topography</b>  |  '
                            f'{subject}  |  <span style="color:{"#f87171" if alcoholic else "#34d399"}">'
                            f'{class_lbl}</span>  |  t = {t_ms} ms')
            )
        ))
    fig.frames = frames

    # ── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f'<b>Brain EEG Topography</b>  |  {subject}  |  '
                  f'<span style="color:{"#f87171" if alcoholic else "#34d399"}">'
                  f'{class_lbl}</span>  |  t = 0 ms'),
            x=0.5, font=dict(color='#e2e8f0', size=17, family='Arial')
        ),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        scene=dict(
            bgcolor='#0f172a',
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=0.0, y=0.1, z=2.2),
                up=dict(x=0, y=1, z=0),
            ),
        ),
        legend=dict(
            title=dict(text='Brain Lobes', font=dict(color='#94a3b8', size=11)),
            font=dict(color='#cbd5e1', size=10),
            bgcolor='rgba(15,23,42,0.8)',
            bordercolor='#334155',
            borderwidth=1,
            x=0.01, y=0.97,
        ),
        margin=dict(l=0, r=0, t=56, b=0),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(color='#e2e8f0', size=12),
            x=0.5, xanchor='center', y=-0.07, yanchor='top',
            buttons=[
                dict(label='  Play  ',
                     method='animate',
                     args=[None, dict(
                         frame=dict(duration=80, redraw=True),
                         fromcurrent=True,
                         mode='immediate',
                         transition=dict(duration=0)
                     )]),
                dict(label='  Pause  ',
                     method='animate',
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode='immediate',
                         transition=dict(duration=0)
                     )]),
                dict(label='  Loop  ',
                     method='animate',
                     args=[None, dict(
                         frame=dict(duration=80, redraw=True),
                         fromcurrent=False,
                         mode='immediate',
                         transition=dict(duration=0)
                     )]),
            ],
        )],
        sliders=[dict(
            active=0,
            pad=dict(t=10, b=10),
            bgcolor='#1e293b',
            bordercolor='#334155',
            font=dict(color='#94a3b8', size=10),
            currentvalue=dict(
                prefix='Frame: ',
                font=dict(color='#60a5fa', size=12),
                visible=True, xanchor='center'
            ),
            steps=[dict(
                method='animate',
                args=[[str(fi)], dict(
                    mode='immediate', transition=dict(duration=0),
                    frame=dict(duration=0, redraw=True)
                )],
                label=str(fi)
            ) for fi in range(n_anim_frames)],
            x=0.05, len=0.90,
        )],
    )

    return fig


# ==================================================================
# CUSTOM HTML TEMPLATE
# ==================================================================
HTML_WRAPPER = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Brain EEG Topography Viewer</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0;
      background: #020617;
      font-family: 'Segoe UI', Arial, sans-serif;
      color: #e2e8f0;
      overflow: hidden;
    }}
    #header {{
      position: fixed; top: 0; left: 0; right: 0; height: 52px;
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
      border-bottom: 1px solid #334155;
      display: flex; align-items: center; padding: 0 20px;
      z-index: 999; gap: 16px;
    }}
    #header h1 {{
      font-size: 17px; font-weight: 700; margin: 0;
      background: linear-gradient(90deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .badge {{
      font-size: 11px; padding: 3px 10px; border-radius: 20px;
      font-weight: 600; letter-spacing: 0.5px;
    }}
    .badge-alc {{ background: rgba(239,68,68,0.15); color: #f87171;
                  border: 1px solid rgba(239,68,68,0.3); }}
    .badge-ctl {{ background: rgba(16,185,129,0.15); color: #34d399;
                  border: 1px solid rgba(16,185,129,0.3); }}
    .sep {{ width: 1px; height: 24px; background: #334155; }}
    .meta {{ font-size: 12px; color: #64748b; }}
    #info-panel {{
      position: fixed; top: 60px; right: 10px; width: 210px;
      background: rgba(15,23,42,0.92);
      border: 1px solid #334155; border-radius: 12px;
      padding: 14px; z-index: 998;
      backdrop-filter: blur(12px);
    }}
    #info-panel h3 {{
      margin: 0 0 10px; font-size: 12px; color: #60a5fa;
      text-transform: uppercase; letter-spacing: 1px;
    }}
    .lobe-row {{
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 7px; font-size: 12px;
    }}
    .lobe-dot {{
      width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0;
    }}
    #controls-hint {{
      position: fixed; bottom: 80px; left: 10px; width: 190px;
      background: rgba(15,23,42,0.88);
      border: 1px solid #334155; border-radius: 12px;
      padding: 12px; font-size: 11px; color: #64748b;
      backdrop-filter: blur(8px);
    }}
    #controls-hint h3 {{
      margin: 0 0 8px; font-size: 11px; color: #60a5fa;
      text-transform: uppercase; letter-spacing: 1px;
    }}
    #controls-hint p {{ margin: 4px 0; }}
    #plot-wrapper {{
      position: fixed; top: 52px; left: 0; right: 0; bottom: 0;
    }}
  </style>
</head>
<body>

<div id="header">
  <h1>Brain EEG Topography</h1>
  <div class="sep"></div>
  <span class="badge {badge_class}">{class_lbl}</span>
  <span class="meta">{subject}</span>
  <div class="sep"></div>
  <span class="meta">256 Hz &nbsp;|&nbsp; 64 ch &nbsp;|&nbsp; 1 sec epoch</span>
</div>

<div id="info-panel">
  <h3>Brain Regions</h3>
  <div class="lobe-row"><div class="lobe-dot" style="background:#3b82f6"></div>Frontal Lobe</div>
  <div class="lobe-row"><div class="lobe-dot" style="background:#a855f7"></div>Central / Motor</div>
  <div class="lobe-row"><div class="lobe-dot" style="background:#10b981"></div>Parietal Lobe</div>
  <div class="lobe-row"><div class="lobe-dot" style="background:#f59e0b"></div>Temporal Lobes</div>
  <div class="lobe-row"><div class="lobe-dot" style="background:#ef4444"></div>Occipital Lobe</div>
  <hr style="border-color:#334155; margin:10px 0">
  <h3 style="color:#94a3b8">EEG Scale</h3>
  <div class="lobe-row"><div class="lobe-dot" style="background:#1a1aff"></div>Neg. Voltage</div>
  <div class="lobe-row"><div class="lobe-dot" style="background:#ffffff"></div>Zero / Rest</div>
  <div class="lobe-row"><div class="lobe-dot" style="background:#cc0000"></div>Pos. Voltage</div>
</div>

<div id="controls-hint">
  <h3>Controls</h3>
  <p>Left drag -> Rotate</p>
  <p>Scroll wheel -> Zoom</p>
  <p>Right drag -> Pan</p>
  <p>Hover dots -> Electrode</p>
  <p>Play button -> Animate</p>
</div>

<div id="plot-wrapper">
{plot_html}
</div>

</body>
</html>"""


# ==================================================================
# MAIN
# ==================================================================
def main():
    TRAIN_DIR = 'SMNI_CMI_TRAIN'
    if not os.path.exists(TRAIN_DIR):
        print(f'[ERROR] "{TRAIN_DIR}" not found. Run from Internship folder.')
        return

    all_files = [os.path.join(r, f)
                 for r, _, fs in os.walk(TRAIN_DIR)
                 for f in fs if f.endswith('.gz')]
    if not all_files:
        print('[ERROR] No .gz EEG files found.')
        return

    print(f'\n  Found {len(all_files)} EEG trial files.')
    filepath = pick_file(all_files)

    fname     = os.path.basename(filepath)
    subject   = os.path.basename(os.path.dirname(filepath))
    alcoholic = fname.startswith('co2a')
    class_lbl = 'ALCOHOLIC' if alcoholic else 'CONTROL'

    # Build figure
    fig = build_figure(filepath, n_anim_frames=40)

    # Export to standalone HTML
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config=dict(
            displayModeBar=True,
            modeBarButtonsToRemove=['lasso2d', 'select2d'],
            displaylogo=False,
            responsive=True
        )
    )

    # Inject into premium wrapper
    out_html = HTML_WRAPPER.format(
        badge_class='badge-alc' if alcoholic else 'badge-ctl',
        class_lbl=class_lbl,
        subject=subject,
        plot_html=plot_html,
    )

    # Fix plotly div to fill container
    out_html = out_html.replace(
        '<div>',
        '<div style="width:100%;height:100%;">',
        1   # only first plotly div
    )

    # Save
    safe_name = fname.replace('.gz', '').replace('.', '_')
    out_path  = f'brain_topo_{safe_name}.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out_html)

    print(f'\n[SUCCESS] Saved: {os.path.abspath(out_path)}')
    print('Open the HTML file in Chrome / Edge / Firefox.\n')
    print('  Controls:')
    print('  - Left-click + drag  ->  Rotate the 3D head')
    print('  - Scroll wheel       ->  Zoom in / out')
    print('  - Right-click + drag ->  Pan')
    print('  - Play button        ->  Animate brain activity (1 second)')
    print('  - Slider             ->  Scrub to any time sample')
    print('  - Hover black dots   ->  See electrode names')
    print()
    print('  Color key (Lobes):')
    print('  - BLUE   = Frontal    (executive function, decision)')
    print('  - PURPLE = Central    (motor strip, movement)')
    print('  - GREEN  = Parietal   (spatial, sensory integration)')
    print('  - AMBER  = Temporal   (memory, language, hearing)')
    print('  - RED    = Occipital  (visual processing)')
    print()
    print('  Color key (EEG):')
    print('  - Deep RED  = strong positive voltage (excitation)')
    print('  - Deep BLUE = strong negative voltage (inhibition)')
    print('  - WHITE     = near zero (resting baseline)')


if __name__ == '__main__':
    main()
