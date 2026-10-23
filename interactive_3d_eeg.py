import os
import gzip
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# =========================================================
# CONFIGURATION
# =========================================================
TRAIN_DIR = "SMNI_CMI_TRAIN"
NUM_OPTIONS = 4

# =========================================================
# EEG FILE PARSER
# =========================================================
def read_eeg_file(filepath):
    """Parses a raw EEG .gz file into a (64 channels x 256 time) matrix."""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f
                 if not line.startswith('#') and line.strip()]

    columns = ['trial_number', 'sensor_position', 'sample_num', 'sensor_value']
    df = pd.DataFrame(lines, columns=columns)
    df['sensor_value'] = df['sensor_value'].astype(float)
    df['sample_num']   = df['sample_num'].astype(int)

    matrix       = df.pivot(index='sensor_position',
                             columns='sample_num',
                             values='sensor_value').values
    channel_names = df['sensor_position'].unique().tolist()
    return matrix, channel_names

# =========================================================
# SELECTION MENU
# =========================================================
def pick_file(all_files):
    """
    Randomly samples 4 files, prints a numbered menu,
    and returns the one the user chooses.
    """
    # Pick 4 unique random files (or fewer if dataset is tiny)
    k = min(NUM_OPTIONS, len(all_files))
    choices = random.sample(all_files, k)

    # Determine class label from filename (co2a = Alcoholic, co2c = Control)
    def label(path):
        fname = os.path.basename(path)
        if fname.startswith('co2a'):
            return "ALCOHOLIC"
        elif fname.startswith('co2c'):
            return "CONTROL  "
        return "UNKNOWN  "

    print("\n" + "=" * 60)
    print("   INTERACTIVE 3D EEG VIEWER  —  File Selection")
    print("=" * 60)
    print("  4 random EEG trials have been selected for you.\n")

    for i, f in enumerate(choices, start=1):
        fname   = os.path.basename(f)
        subject = os.path.basename(os.path.dirname(f))
        lbl     = label(f)
        print(f"  [{i}]  {fname}")
        print(f"       Subject : {subject}")
        print(f"       Class   : {lbl}")
        print()

    print("-" * 60)

    while True:
        try:
            raw = input(f"  Enter your choice (1 – {k}): ").strip()
            idx = int(raw) - 1
            if 0 <= idx < k:
                selected = choices[idx]
                print(f"\n  Selected: {os.path.basename(selected)}\n")
                return selected
            else:
                print(f"  Please enter a number between 1 and {k}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")

# =========================================================
# 3D PLOT BUILDER
# =========================================================
def build_and_save(filepath):
    """Loads, filters, and generates the animated 3D Plotly figure."""

    print("Loading and filtering EEG data ...")
    data, channel_names = read_eeg_file(filepath)

    # 4th-order zero-phase Butterworth bandpass (0.5 – 100 Hz)
    fs = 256
    b, a = butter(4, [0.5 / (fs / 2), 100.0 / (fs / 2)], btype='band')
    filt = filtfilt(b, a, data, axis=-1)

    print("Building animation frames (this takes a few seconds) ...")

    n_channels, n_time = filt.shape
    channels = np.arange(n_channels)
    time     = np.arange(n_time)
    z_min, z_max = float(np.min(filt)), float(np.max(filt))

    # Determine colorscale: blue-ish for Control, warm for Alcoholic
    fname = os.path.basename(filepath)
    colorscale = 'Plasma' if fname.startswith('co2a') else 'Viridis'
    class_label = "ALCOHOLIC" if fname.startswith('co2a') else "CONTROL"

    # --- Animation frames (reveal signal left-to-right) ---
    step = 8
    frames = []
    for t in range(step, n_time + 1, step):
        z_frame = np.copy(filt)
        z_frame[:, t:] = np.nan
        frames.append(go.Frame(
            data=[go.Surface(z=z_frame, x=time, y=channels,
                             colorscale=colorscale,
                             cmin=z_min, cmax=z_max,
                             showscale=False)],
            name=str(t)
        ))

    # Initial frame
    z_init = np.copy(filt)
    z_init[:, step:] = np.nan

    subject = os.path.basename(os.path.dirname(filepath))
    title   = (f"3D EEG Spatio-Temporal Topology  |  "
               f"Subject: {subject}  |  Class: {class_label}  |  File: {fname}")

    fig = go.Figure(
        data=[go.Surface(
            z=z_init, x=time, y=channels,
            colorscale=colorscale,
            cmin=z_min, cmax=z_max,
            colorbar=dict(title='Amplitude (uV)', thickness=18)
        )]
    )

    # --- Layout ---
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        width=1300, height=820,
        scene=dict(
            xaxis_title='Time (samples, 256 Hz)',
            yaxis_title='EEG Channel (1 – 64)',
            zaxis_title='Amplitude (uV)',
            xaxis=dict(range=[0, n_time - 1]),
            yaxis=dict(range=[0, n_channels - 1],
                       ticktext=channel_names,
                       tickvals=list(channels)),
            zaxis=dict(range=[z_min, z_max]),
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.3))
        ),
        # Play / Pause buttons
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.12, y=0.0,
            xanchor="right", yanchor="top",
            pad=dict(t=87, r=10),
            buttons=[
                dict(
                    label="  Play  ",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=90, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0)
                    )]
                ),
                dict(
                    label="  Pause  ",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0)
                    )]
                )
            ]
        )],
        # Time scrub slider
        sliders=[dict(
            active=0,
            yanchor="top", xanchor="left",
            currentvalue=dict(
                font=dict(size=13),
                prefix='Time step: ',
                visible=True
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.88, x=0.12, y=0,
            steps=[
                dict(
                    method='animate',
                    args=[[str(t)], dict(
                        mode='immediate',
                        frame=dict(duration=90, redraw=True),
                        transition=dict(duration=0)
                    )],
                    label=str(t)
                )
                for t in range(step, n_time + 1, step)
            ]
        )]
    )

    fig.frames = frames

    # --- Save ---
    safe_name = fname.replace('.gz', '').replace('.', '_')
    html_path = f"interactive_3d_eeg_{safe_name}.html"
    fig.write_html(html_path)

    print("\n[SUCCESS] Plot saved!")
    print(f"  File : {os.path.abspath(html_path)}")
    print("  Open this HTML file in Chrome / Edge / Firefox.")
    print("  - Drag to rotate   |  Scroll to zoom   |  Right-drag to pan")
    print("  - Press 'Play' to animate the waveform across time")
    print("  - Use the slider to scrub to any moment manually")

# =========================================================
# MAIN
# =========================================================
def main():
    # 1. Collect all .gz files recursively
    if not os.path.exists(TRAIN_DIR):
        print(f"[ERROR] Directory '{TRAIN_DIR}' not found. "
              f"Run this script from your Internship folder.")
        return

    all_files = []
    for root, _, filenames in os.walk(TRAIN_DIR):
        for fname in filenames:
            if fname.endswith('.gz'):
                all_files.append(os.path.join(root, fname))

    if not all_files:
        print(f"[ERROR] No .gz EEG files found under '{TRAIN_DIR}'.")
        return

    print(f"\n  Found {len(all_files)} EEG trial files in total.")

    # 2. Let the user pick one of 4 random options
    selected = pick_file(all_files)

    # 3. Build and save the 3D animated plot
    build_and_save(selected)


if __name__ == "__main__":
    main()
