"""
FILE: 02_dsp_filtering_mne.py
PROJECT: Alcoholic EEG Classification - Internship Research
PAPER: Farsi et al. (2021) - Classification of Alcoholic EEG Signals
PURPOSE: Digital Signal Processing pipeline.
         Implements both IIR (Butterworth) and FIR (MNE) bandpass filtering
         as described in the preprocessing stage of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
from scipy.signal import butter, filtfilt, welch
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\Visha\Desktop\Internshhip"
OUT_DIR    = r"C:\Users\Visha\Desktop\Internshhip\outputs"

FS         = 256     # Sampling frequency (Hz)
LOWCUT     = 0.5     # Hz — remove DC drift and slow baseline wander
HIGHCUT    = 45.0    # Hz — remove high-frequency muscle (EMG) artifacts
NOTCH_FREQ = 45.0    # Hz — remove power-line interference
FILT_ORDER = 4
# ────────────────────────────────────────────────────────────


def apply_butterworth_bandpass(
    X: np.ndarray,
    fs: int = FS,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    order: int = FILT_ORDER
) -> np.ndarray:
    """
    Zero-phase IIR Butterworth bandpass filter.
    Applied along time axis (axis=2) for tensor (N, C, T).

    Zero-phase (filtfilt) eliminates group delay — critical for EEG.
    """
    nyquist = 0.5 * fs
    low  = lowcut  / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, X, axis=2).astype(np.float32)


def apply_mne_fir_filter(
    X: np.ndarray,
    fs: int = FS,
    l_freq: float = LOWCUT,
    h_freq: float = HIGHCUT
) -> np.ndarray:
    """
    FIR filter using MNE-Python's robust implementation.
    FIR avoids the phase non-linearity of IIR filters.
    Preferred approach for research-grade EEG processing.
    """
    ch_names = [f'EEG_{i:03d}' for i in range(X.shape[1])]
    ch_types = ['eeg'] * X.shape[1]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # MNE EpochsArray expects (n_epochs, n_channels, n_times)
    epochs = mne.EpochsArray(X, info, verbose=False)
    epochs.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        fir_design='firwin',
        verbose=False
    )
    return epochs.get_data().astype(np.float32)


def compute_psd(signal: np.ndarray, fs: int = FS) -> tuple:
    """Compute Welch's Power Spectral Density for a single-channel signal."""
    freqs, psd = welch(signal, fs=fs, nperseg=128)
    return freqs, psd


def plot_filtering_comparison(
    raw: np.ndarray,
    filtered: np.ndarray,
    channel_idx: int = 0,
    trial_idx: int = 0,
    fs: int = FS,
    save_path: str = None
) -> None:
    """
    Plots a 4-panel comparison figure:
    [Raw Time | Filtered Time | Raw PSD | Filtered PSD]
    """
    time_axis = np.arange(raw.shape[2]) / fs

    fig = plt.figure(figsize=(16, 8), facecolor="#f9fafc18")
    fig.suptitle(
        f'DSP Filtering Comparison  |  Channel {channel_idx}  |  Trial {trial_idx}',
        color='white', fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    raw_sig  = raw[trial_idx, channel_idx, :]
    filt_sig = filtered[trial_idx, channel_idx, :]
    f_raw,  p_raw  = compute_psd(raw_sig,  fs)
    f_filt, p_filt = compute_psd(filt_sig, fs)

    plot_cfg = [
        (gs[0, 0], time_axis, raw_sig,  "#8f2525", 'Raw EEG Signal',        'Time (s)',    'Amplitude (µV)'),
        (gs[0, 1], time_axis, filt_sig, "#0d737a", 'Filtered EEG Signal',   'Time (s)',    'Amplitude (µV)'),
        (gs[1, 0], f_raw,     p_raw,    "#79561a", 'PSD — Raw',             'Frequency (Hz)', 'Power (µV²/Hz)'),
        (gs[1, 1], f_filt,    p_filt,   "#118362", 'PSD — Filtered (FIR)',  'Frequency (Hz)', 'Power (µV²/Hz)'),
    ]

    for spec, x, y, color, title, xlabel, ylabel in plot_cfg:
        ax = fig.add_subplot(spec)
        ax.plot(x, y, color=color, linewidth=1.2)
        ax.set_facecolor("#7a8ab3")
        ax.set_title(title, color='white', fontsize=11)
        ax.set_xlabel(xlabel, color='#94a8c4', fontsize=9)
        ax.set_ylabel(ylabel, color='#94a8c4', fontsize=9)
        ax.tick_params(colors='#5d7499')
        ax.spines[:].set_color('#1e2d45')
        if 'PSD' in title:
            ax.set_xlim([0, 80])
            ax.axvspan(LOWCUT, HIGHCUT, alpha=0.1, color='cyan', label='Passband')
            ax.legend(fontsize=8, facecolor='#0d1526', labelcolor='white')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    plt.show()
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("  DSP Filtering Pipeline")
    print("=" * 60)

    data_path = Path(DATA_DIR)
    out_path  = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    print(f"Loaded X_train: {X_train.shape}")

    # Step 1: IIR Butterworth
    print("\nApplying Butterworth IIR filter...")
    X_butter = apply_butterworth_bandpass(X_train)
    print(f"  IIR filtered shape: {X_butter.shape}")

    # Step 2: MNE FIR (research-grade)
    print("Applying MNE FIR filter...")
    X_fir = apply_mne_fir_filter(X_train)
    print(f"  FIR filtered shape: {X_fir.shape}")

    # Save filtered tensors
    np.save(str(data_path / 'X_filtered_butter.npy'), X_butter)
    np.save(str(data_path / 'X_filtered_fir.npy'),    X_fir)
    print("Filtered tensors saved.")

    # Visualize
    plot_filtering_comparison(
        X_train, X_fir,
        channel_idx=0,
        trial_idx=0,
        save_path=str(out_path / 'dsp_comparison.png')
    )

    print("\nDSP pipeline complete. Proceed to 03_algorithm1_pca_ann.py")
