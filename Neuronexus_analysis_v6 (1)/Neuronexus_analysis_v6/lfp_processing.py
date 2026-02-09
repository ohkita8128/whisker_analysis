"""
processing.py - 前処理関数
フィルタ、チャンネル処理、モーション解析、ICA、環境ノイズ除去
"""
import numpy as np
from scipy import signal
from scipy.ndimage import binary_dilation


def estimate_fir_taps(fs, lowcut, highcut, transition_width=None):
    """
    FIRフィルターの適切なタップ数を自動計算
    
    Parameters
    ----------
    fs : int
        サンプリングレート
    lowcut, highcut : float
        バンドパスの下限・上限周波数 (Hz)
    transition_width : float or None
        遷移帯域幅 (Hz)。Noneの場合は自動推定
    
    Returns
    -------
    numtaps : int
        推奨タップ数（奇数）
    """
    if transition_width is None:
        # 遷移帯域幅を自動推定
        # lowcutの50%か1Hzの大きい方
        transition_width = max(lowcut * 0.5, 1.0)
    
    # Kaiser窓の経験式: numtaps ≈ 3.3 * fs / transition_width
    numtaps = int(3.3 * fs / transition_width)
    
    # 奇数に（FIRバンドパスの要件）
    if numtaps % 2 == 0:
        numtaps += 1
    
    # 最小値を保証
    numtaps = max(numtaps, 101)
    
    return numtaps


def bandpass_notch_filter(data, lowcut, highcut, fs, 
                          filter_type='iir', order=4, fir_numtaps=None,
                          notch_freq=None, notch_Q=30):
    """
    バンドパス + ノッチフィルタ
    
    Parameters
    ----------
    data : ndarray
        入力データ (samples, channels)
    lowcut, highcut : float
        バンドパスの下限・上限周波数 (Hz)
    fs : int
        サンプリングレート
    filter_type : str
        'iir' (Butterworth) or 'fir' (FIR窓関数法)
    order : int
        IIRフィルタの次数 (デフォルト: 4)
    fir_numtaps : int or None
        FIRフィルタのタップ数。Noneの場合は自動計算
    notch_freq : float or None
        ノッチフィルタ周波数 (Hz), Noneで無効
    notch_Q : float
        ノッチフィルタのQ値 (デフォルト: 30)
    
    Returns
    -------
    filtered : ndarray
    actual_taps : int or None
        実際に使用したFIRタップ数（IIRの場合はNone）
    """
    nyq = 0.5 * fs
    actual_taps = None
    
    # === バンドパスフィルタ ===
    if filter_type.lower() == 'fir':
        # タップ数を自動計算または指定値を使用
        if fir_numtaps is None or fir_numtaps <= 0:
            fir_numtaps = estimate_fir_taps(fs, lowcut, highcut)
        
        # タップ数は奇数に
        if fir_numtaps % 2 == 0:
            fir_numtaps += 1
        
        actual_taps = fir_numtaps
        
        # FIRフィルタ（窓関数法）
        b = signal.firwin(fir_numtaps, [lowcut/nyq, highcut/nyq], pass_zero='bandpass')
        filtered = signal.filtfilt(b, [1.0], data, axis=0)
    else:
        # IIRフィルタ（Butterworth）
        sos = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='bandpass', output='sos')
        filtered = signal.sosfiltfilt(sos, data, axis=0)
    
    # === ノッチフィルタ（IIR固定）===
    if notch_freq is not None:
        b, a = signal.iirnotch(notch_freq, notch_Q, fs)
        filtered = signal.filtfilt(b, a, filtered, axis=0)
    
    return filtered, actual_taps


def get_frame_times(events, min_count=1000, max_interval=1.0, verbose=True):
    """フレーム同期イベントを取得"""
    for evt in events:
        if len(evt.times) >= min_count:
            times = np.array(evt.times)
            intervals = np.diff(times)
            large_gaps = np.where(intervals > max_interval)[0]
            if len(large_gaps) > 0 and large_gaps[0] < 10:
                times = times[large_gaps[0] + 1:]
            if verbose:
                print(f"  フレーム同期: {evt.name}, {len(times)}個")
            return times
    raise ValueError("フレーム同期イベントが見つかりません")


def detect_bad_channels(lfp_data, channel_order, threshold=3.0, verbose=True):
    """分散が異常なチャンネルを検出"""
    variances = np.var(lfp_data, axis=0)
    median_var = np.median(variances)
    mad = np.median(np.abs(variances - median_var))
    upper = median_var + threshold * mad * 1.4826
    lower = median_var - threshold * mad * 1.4826
    bad_high = np.where(variances > upper)[0]
    bad_low = np.where(variances < lower)[0]
    bad_channels = list(bad_high) + list(bad_low)
    
    if verbose and bad_channels:
        for i in bad_channels:
            print(f"    異常チャンネル: D{i}(Ch{channel_order[i]}), var={variances[i]:.6f}")
    return bad_channels, variances


def analyze_video_motion(video_path, roi=None, threshold=15, blur=5):
    """動画からモーション量を計算"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if roi is None:
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_bright = cv2.convertScaleAbs(frame, alpha=2.0, beta=50)
        print("  ROIを選択 (Enter確定, C取消)")
        roi = cv2.selectROI("Select ROI", frame_bright, fromCenter=False)
        cv2.destroyAllWindows()
    
    x, y, w, h = roi
    prev_gray = None
    motion_values = []
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh_img = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            motion_values.append(np.sum(thresh_img > 0))
        else:
            motion_values.append(0)
        prev_gray = gray
    cap.release()
    return np.array(motion_values), roi


def create_noise_mask(motion, frame_times, lfp_times, fs, percentile=75, expand_sec=0.1):
    """モーションからノイズマスク作成"""
    n = min(len(motion), len(frame_times))
    motion_resampled = np.interp(lfp_times, frame_times[:n], motion[:n])
    threshold = np.percentile(motion_resampled, percentile)
    mask = motion_resampled > threshold
    expand_samples = int(expand_sec * fs)
    mask = binary_dilation(mask, iterations=expand_samples)
    return mask, motion_resampled, threshold


def remove_artifact_ica(lfp_data, noise_mask, n_components=None, 
                        noise_ratio_threshold=2.5, max_remove=3, verbose=True):
    """ICAでモーションアーティファクト除去"""
    from sklearn.decomposition import FastICA
    
    lfp_mean = np.mean(lfp_data, axis=0)
    lfp_std = np.std(lfp_data, axis=0)
    lfp_std[lfp_std == 0] = 1
    lfp_norm = (lfp_data - lfp_mean) / lfp_std
    
    if n_components is None:
        n_components = lfp_data.shape[1]
    
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=0.01)
    sources = ica.fit_transform(lfp_norm)
    
    # ノイズ成分特定
    noise_ratios = []
    for i in range(n_components):
        var_noise = np.var(sources[noise_mask, i])
        var_clean = np.var(sources[~noise_mask, i])
        ratio = var_noise / var_clean if var_clean > 0 else np.inf
        noise_ratios.append(ratio)
    
    sorted_idx = np.argsort(noise_ratios)[::-1]
    remove_idx = []
    for i in sorted_idx:
        if noise_ratios[i] > noise_ratio_threshold and len(remove_idx) < max_remove:
            remove_idx.append(i)
            if verbose:
                print(f"    IC{i}: ratio={noise_ratios[i]:.2f} → 除去")
    
    sources_clean = sources.copy()
    sources_clean[:, remove_idx] = 0
    lfp_reconstructed = ica.inverse_transform(sources_clean) * lfp_std + lfp_mean
    return lfp_reconstructed, remove_idx, noise_ratios, sources


# ============================================================
# 環境ノイズ除去
# ============================================================

def load_noise_reference(noise_file, channel_order, fs_expected=1000, verbose=True):
    """
    環境ノイズ記録ファイルを読み込み
    
    Parameters
    ----------
    noise_file : str
        ノイズ記録の.plxファイルパス
    channel_order : list
        チャンネル並び順
    fs_expected : int
        期待するサンプリングレート
    
    Returns
    -------
    noise_lfp : ndarray (samples, channels)
    fs : int
    """
    import neo
    
    if verbose:
        print(f"  環境ノイズファイル: {noise_file}")
    
    plx = neo.io.PlexonIO(filename=noise_file)
    data = plx.read()
    seg = data[0].segments[0]
    
    noise_raw = np.array(seg.analogsignals[1][:, channel_order])
    fs = int(seg.analogsignals[1].sampling_rate)
    
    if fs != fs_expected and verbose:
        print(f"    Warning: サンプリングレートが異なります ({fs}Hz vs {fs_expected}Hz)")
    
    if verbose:
        print(f"    ノイズデータ: {noise_raw.shape}, {fs}Hz")
    
    return noise_raw, fs


def detect_noise_peaks(noise_data, fs, threshold_db=10, freq_min=5, freq_max=100, 
                       max_peaks=10, verbose=True):
    """
    環境ノイズ記録からピーク周波数を検出
    
    Parameters
    ----------
    noise_data : ndarray (samples, channels) or (samples,)
        環境ノイズ記録
    fs : int
        サンプリングレート
    threshold_db : float
        ピーク検出閾値（中央値からのdB差）
    freq_min, freq_max : float
        ピーク検出する周波数範囲
    max_peaks : int
        検出する最大ピーク数
    
    Returns
    -------
    peak_freqs : list
        検出されたピーク周波数
    noise_psd : tuple (freqs, psd)
        ノイズのパワースペクトル
    """
    from scipy.signal import find_peaks, welch
    
    # 1チャンネルに平均化
    if noise_data.ndim > 1:
        noise_1d = noise_data.mean(axis=1)
    else:
        noise_1d = noise_data
    
    # PSD計算
    nperseg = min(fs * 2, len(noise_1d) // 2)
    freqs, psd = welch(noise_1d, fs, nperseg=nperseg)
    
    # 周波数範囲でマスク
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_masked = freqs[freq_mask]
    psd_masked = psd[freq_mask]
    
    # dBスケールに変換
    psd_db = 10 * np.log10(psd_masked + 1e-10)
    median_db = np.median(psd_db)
    
    # ピーク検出
    peaks, props = find_peaks(psd_db - median_db, prominence=threshold_db)
    
    if len(peaks) == 0:
        if verbose:
            print("    ノイズピークが検出されませんでした")
        return [], (freqs, psd)
    
    # 上位のピークを選択
    if len(peaks) > max_peaks:
        top_idx = np.argsort(props['prominences'])[::-1][:max_peaks]
        peaks = peaks[top_idx]
    
    peak_freqs = sorted(freqs_masked[peaks])
    
    if verbose:
        print(f"    検出ピーク ({len(peak_freqs)}個): {[f'{f:.1f}Hz' for f in peak_freqs]}")
    
    return peak_freqs, (freqs, psd)


def remove_noise_by_notch(lfp_data, peak_freqs, fs, Q=30, verbose=True):
    """
    検出されたピーク周波数にノッチフィルタを適用
    
    Parameters
    ----------
    lfp_data : ndarray (samples, channels)
        実験データ
    peak_freqs : list
        除去する周波数のリスト
    fs : int
        サンプリングレート
    Q : float
        ノッチフィルタのQ値（大きいほど狭い帯域）
    
    Returns
    -------
    filtered : ndarray
        ノイズ除去後のデータ
    """
    from scipy.signal import iirnotch, filtfilt
    
    filtered = lfp_data.copy()
    
    for freq in peak_freqs:
        if 1 < freq < fs/2 - 1:
            b, a = iirnotch(freq, Q, fs)
            filtered = filtfilt(b, a, filtered, axis=0)
    
    if verbose:
        print(f"    {len(peak_freqs)}個の周波数にノッチフィルタ適用")
    
    return filtered




def remove_environmental_noise(lfp_data, noise_data, fs, 
                                threshold_db=10, Q=30, max_peaks=10,
                                freq_min=5, freq_max=100, verbose=True):
    """
    環境ノイズ除去のメイン関数（検出＋除去）
    
    Parameters
    ----------
    lfp_data : ndarray (samples, channels)
        実験データ
    noise_data : ndarray (samples, channels)
        環境ノイズ記録
    fs : int
        サンプリングレート
    threshold_db : float
        ピーク検出閾値（dB）
    Q : float
        ノッチフィルタのQ値
    max_peaks : int
        除去する最大ピーク数
    freq_min, freq_max : float
        検出する周波数範囲
    
    Returns
    -------
    filtered : ndarray
        ノイズ除去後のデータ
    peak_freqs : list
        検出・除去したピーク周波数
    noise_psd : tuple (freqs, psd)
        ノイズのパワースペクトル（プロット用）
    """
    if verbose:
        print("  環境ノイズ除去:")
    
    # ピーク検出
    peak_freqs, noise_psd = detect_noise_peaks(
        noise_data, fs,
        threshold_db=threshold_db,
        freq_min=freq_min,
        freq_max=freq_max,
        max_peaks=max_peaks,
        verbose=verbose
    )
    
    if len(peak_freqs) == 0:
        return lfp_data, [], noise_psd
    
    # ノッチフィルタ適用
    filtered = remove_noise_by_notch(lfp_data, peak_freqs, fs, Q=Q, verbose=verbose)
    
    return filtered, peak_freqs, noise_psd

def remove_known_harmonics(lfp_data, fs, fundamental=10, n_harmonics=5, Q=30, verbose=True):
    """
    既知の高調波ノイズを除去（10Hz矩形波 → 奇数倍高調波）
    
    Parameters
    ----------
    fundamental : float
        基本周波数（Hz）
    n_harmonics : int
        除去する高調波の数
    """
    from scipy.signal import iirnotch, filtfilt
    
    # 奇数倍高調波: 10, 30, 50, 70, 90 Hz
    harmonics = [fundamental * (2*i + 1) for i in range(n_harmonics)]
    
    if verbose:
        print(f"  高調波ノイズ除去: {harmonics} Hz")
    
    filtered = lfp_data.copy()
    for freq in harmonics:
        if freq < fs / 2:  # ナイキスト周波数以下のみ
            b, a = iirnotch(freq, Q, fs)
            filtered = filtfilt(b, a, filtered, axis=0)
    
    return filtered, harmonics

# ============================================================
# 刺激イベント関連
# ============================================================

def get_stim_events(events, session_name="EVT02", stim_name="EVT01", verbose=True):
    """刺激イベントを取得"""
    session_times, stim_times = None, None
    for evt in events:
        if len(evt.times) > 0:
            if evt.name == session_name:
                session_times = np.array(evt.times)
            if evt.name == stim_name:
                stim_times = np.array(evt.times)
    if verbose:
        print(f"  刺激イベント: {len(stim_times) if stim_times is not None else 0}個")
    return session_times, stim_times


def create_stim_mask(lfp_times, session_ranges, margin=0.1):
    """刺激マスク作成"""
    mask = np.zeros(len(lfp_times), dtype=bool)
    for start, end in session_ranges:
        mask |= (lfp_times >= start) & (lfp_times <= end + margin)
    return mask


# ============================================================
# パワースペクトル解析
# ============================================================

def compute_psd(data, mask, fs, nperseg=1024):
    """パワースペクトル密度を計算"""
    freqs, psd = signal.welch(data[mask], fs=fs, nperseg=nperseg, axis=0)
    return freqs, np.mean(psd, axis=1), np.std(psd, axis=1) / np.sqrt(psd.shape[1])


def compute_band_power(psd, freqs, band):
    """特定バンドのパワーを計算"""
    mask = (freqs >= band[0]) & (freqs < band[1])
    return np.mean(psd[mask])


# ============================================================
# ウェーブレット解析
# ============================================================

def compute_cwt(lfp_data, fs, freq_min=1, freq_max=100, n_freqs=50, wavelet='cmor1.5-1.0'):
    """
    連続ウェーブレット変換
    
    Parameters
    ----------
    lfp_data : ndarray (n_samples, n_channels)
    fs : int
    freq_min, freq_max : float
    n_freqs : int
    wavelet : str
    
    Returns
    -------
    cwt_power : ndarray (n_freqs, n_channels, n_samples)
    cwt_freqs : ndarray (n_freqs,)
    """
    import pywt
    
    cwt_freqs = np.linspace(freq_min, freq_max, n_freqs)
    scales = pywt.scale2frequency(wavelet, 1) * fs / cwt_freqs
    
    n_samples, n_channels = lfp_data.shape
    cwt_power = np.zeros((n_freqs, n_channels, n_samples))
    
    for ch in range(n_channels):
        coeffs, _ = pywt.cwt(lfp_data[:, ch], scales, wavelet, sampling_period=1/fs)
        cwt_power[:, ch, :] = np.abs(coeffs) ** 2
    
    return cwt_power, cwt_freqs