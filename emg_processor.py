import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
import pywt

FS = 1000
NYQUIST = FS / 2

# --- Filters and transforms ---

def apply_bandpass_filter(signal_data, low_freq=20, high_freq=450, order=4):
    b, a = butter(order, [low_freq / NYQUIST, high_freq / NYQUIST], btype='band')
    return filtfilt(b, a, signal_data)

def apply_notch_filter(signal_data, notch_freq=50, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, FS)
    return filtfilt(b, a, signal_data)

def cwt_denoise(signal_data, wavelet='db4', level=6):
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)

def calculate_moving_rms(signal_data, window_ms=100):
    window_samples = int(window_ms * FS / 1000)
    return pd.Series(signal_data).pow(2).rolling(window=window_samples, center=True).mean().pow(0.5).values

# --- Metrics ---

def detect_contractions_dual_scale(signal_data, fs=FS, threshold_pct=1.5):
    scales_fine = np.arange(1, 32)
    coeffs_fine, _ = pywt.cwt(signal_data, scales_fine, 'cmor1.5-1.0', sampling_period=1/fs)
    energy_fine = np.sum(np.abs(coeffs_fine)**2, axis=0)

    scales_coarse = np.arange(16, 128)
    coeffs_coarse, _ = pywt.cwt(signal_data, scales_coarse, 'cmor1.5-1.0', sampling_period=1/fs)
    energy_coarse = np.sum(np.abs(coeffs_coarse)**2, axis=0)

    from scipy.signal import savgol_filter
    window_length = min(len(energy_coarse), int(0.2 * fs))
    window_length -= (window_length % 2 == 0)
    if window_length >= 3:
        energy_coarse = savgol_filter(energy_coarse, window_length, polyorder=2)

    combined_energy = 0.6 * energy_fine + 0.4 * energy_coarse
    threshold = (threshold_pct / 100.0) * np.max(combined_energy)
    activation_signal = (combined_energy > threshold).astype(int)

    diff_signal = np.diff(activation_signal, prepend=0, append=0)
    starts = np.where(diff_signal == 1)[0]
    ends = np.where(diff_signal == -1)[0] - 1
    min_duration_samples = int(0.05 * fs)
    intervals = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_duration_samples]
    return merge_intervals(intervals, fs, max_gap_ms=150)

def merge_intervals(intervals, fs, max_gap_ms):
    if len(intervals) < 2:
        return intervals
    max_gap = int(max_gap_ms * fs / 1000)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s - le <= max_gap:
            merged[-1] = (ls, e)
        else:
            merged.append((s, e))
    return merged

def calculate_magnitude_metrics(rms, intervals):
    peaks = [np.nanmax(rms[s:e]) for s, e in intervals if e > s]
    avgs = [np.nanmean(rms[s:e]) for s, e in intervals if e > s]
    return np.mean(peaks) if peaks else 0, np.mean(avgs) if avgs else 0

def calculate_frequency_metrics(signal_data, fs, intervals):
    mnfs, mdfs = [], []
    for s, e in intervals:
        seg = signal_data[s:e]
        if len(seg) > 256:
            f, p = welch(seg, fs, nperseg=256)
            mnfs.append(np.sum(f*p)/np.sum(p))
            mdf_idx = np.where(np.cumsum(p) >= np.sum(p)/2)[0][0]
            mdfs.append(f[mdf_idx])
    return np.mean(mnfs) if mnfs else 0, np.mean(mdfs) if mdfs else 0

def calculate_fatigue_index(signal_data, fs, intervals):
    mdfs, times = [], []
    for s, e in intervals:
        seg = signal_data[s:e]
        if len(seg) > 256:
            f, p = welch(seg, fs, nperseg=256)
            mdf_idx = np.where(np.cumsum(p) >= np.sum(p)/2)[0][0]
            mdfs.append(f[mdf_idx])
            times.append((s + e) / (2 * fs))
    if len(mdfs) < 2:
        return 0, [], []
    slope, _ = np.polyfit(times, mdfs, 1)
    return slope, mdfs, times

def calculate_signal_quality_metrics(raw, denoised):
    b, a = butter(4, 1.0/NYQUIST, btype='low')
    drift = np.mean(filtfilt(b, a, raw))
    zcr = np.sum(np.abs(np.diff(np.sign(denoised)))) / (2 * len(denoised))
    return drift, zcr

def calculate_dynamic_metrics(rms, intervals, total_samples, fs):
    ar = sum(e - s for s, e in intervals) / total_samples
    rise, fall = [], []
    for s, e in intervals:
        seg = rms[s:e]
        if len(seg) > 1:
            p = np.nanargmax(seg)
            r = (seg[p] - seg[0]) / (p/fs) if p > 0 else 0
            f = (seg[p] - seg[-1]) / ((len(seg)-p)/fs) if (len(seg) - p) > 0 else 0
            rise.append(r)
            fall.append(f)
    return ar, np.mean(rise) if rise else 0, np.mean(fall) if fall else 0

def calculate_rfd_analog(rms, fs, intervals):
    slopes = []
    for s, e in intervals:
        seg = rms[s:e]
        if len(seg) > 1:
            p = np.nanargmax(seg)
            rise = seg[:p+1]
            if len(rise) > 1:
                s_vals = np.gradient(rise, 1/fs)
                slopes.append(np.nanmax(s_vals))
    return np.mean(slopes) if slopes else 0

def calculate_snr_time_domain(signal_data):
    b, a = butter(4, 400 / NYQUIST, btype='high')
    noise = filtfilt(b, a, signal_data)
    snr = np.mean(signal_data**2) / (np.mean(noise**2) + 1e-12)
    return 10 * np.log10(snr)

def calculate_snr_frequency_domain(signal_data):
    f, p = welch(signal_data, FS, nperseg=min(1024, len(signal_data)//4))
    s_band = (f >= 20) & (f <= 450)
    n_band = (f <= 10) | (f >= 500)
    sp = np.mean(p[s_band])
    np_ = np.mean(p[n_band])
    return 10 * np.log10(sp / (np_ + 1e-12))

# --- Main function ---

def process_emg_dataframe(df: pd.DataFrame):
    results = {}
    output_df = pd.DataFrame({'time': np.arange(len(df)) / FS})

    for col in df.columns:
        if 'emg' not in col.lower():
            continue

        raw = df[col].values
        filtered = apply_bandpass_filter(raw)
        filtered = apply_notch_filter(filtered)
        denoised = cwt_denoise(filtered)
        rms = calculate_moving_rms(denoised)
        intervals = detect_contractions_dual_scale(denoised)

        metrics = {}
        metrics['peak_magnitude'], metrics['avg_magnitude'] = calculate_magnitude_metrics(rms, intervals)
        metrics['fatigue_index'], _, _ = calculate_fatigue_index(denoised, FS, intervals)
        metrics['mean_freq'], metrics['median_freq'] = calculate_frequency_metrics(denoised, FS, intervals)
        metrics['baseline_drift'], metrics['zcr'] = calculate_signal_quality_metrics(raw, denoised)
        metrics['activation_ratio'], metrics['rate_of_rise'], metrics['rate_of_fall'] = calculate_dynamic_metrics(rms, intervals, len(raw), FS)
        metrics['rfd_analog'] = calculate_rfd_analog(rms, FS, intervals)
        metrics['time_snr_raw'] = calculate_snr_time_domain(raw)
        metrics['time_snr_denoised'] = calculate_snr_time_domain(denoised)
        metrics['freq_snr_raw'] = calculate_snr_frequency_domain(raw)
        metrics['freq_snr_denoised'] = calculate_snr_frequency_domain(denoised)

        for key, val in metrics.items():
            results[f"{col}_{key}"] = val

        output_df[f"{col}_filtered"] = denoised
        output_df[f"{col}_rms"] = rms

    return output_df, results
