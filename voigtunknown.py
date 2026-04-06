#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NMR Voigt Integration 
--------------------------------
Hopes and dreams:
- Load Bruker processed spectra from a folder or ZIP archive
- Load known peak regions from Excel
- Detect peaks automatically
- Fit/integrate peaks with Voigt
- Align detected peaks across samples, "fingerprint" 
- Visualize raw spectra, fitted curves, baselines, and integration regions
- Run basic statistics (PCA / PLS-DA)
- Export results to Excel

Future plans:
- Add to PeakNMR to build a larger, more encompassing suite of NMR tools
"""

from __future__ import annotations

import os
import re
import zipfile
import tempfile
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.special import wofz
from sklearn.cluster import DBSCAN
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

import nmrglue as ng

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# types
# -----------------------------------------------------------------------------

@dataclass
class Spectrum:
    ppm: np.ndarray
    intensity: np.ndarray
    sample_name: str
    source_path: str


@dataclass
class PeakResult:
    sample_name: str
    peak_name: Optional[str]
    index: int
    ppm_center: float
    amplitude: float
    sigma: float
    gamma: float
    integral: float
    start_ppm: float
    end_ppm: float
    baseline: float
    fit_success: bool

    @property
    def fwhm_approx(self) -> float:
        return (2.355 * self.sigma + 2.0 * self.gamma) / 2.0


# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------

class Theme:
    BG = "#31363b"
    PANEL = "#3b4148"
    AX_BG = "#23272b"
    FG = "white"
    MUTED = "#c9d1d9"
    PEAK = "#ff6b6b"
    BASELINE = "#f4a261"
    DETECT_FILL = "#2a9d8f"
    KNOWN_FILL = "#457b9d"

    @staticmethod
    def apply_matplotlib() -> None:
        plt.rcParams.update({
            "figure.facecolor": Theme.AX_BG,
            "axes.facecolor": Theme.AX_BG,
            "axes.edgecolor": Theme.FG,
            "axes.labelcolor": Theme.FG,
            "xtick.color": Theme.FG,
            "ytick.color": Theme.FG,
            "text.color": Theme.FG,
            "legend.frameon": True,
        })


def natural_sort_key(text: str) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def extract_zip(zip_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="nmr_zip_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    return tmp_dir


def find_bruker_pdata_dirs(root_dir: str) -> List[str]:
    matches: List[str] = []
    for dirpath, _, _ in os.walk(root_dir):
        p = Path(dirpath)
        if p.name == "1" and p.parent.name == "pdata":
            matches.append(str(p))
    return matches


def load_bruker_spectra(root_dir: str) -> List[Spectrum]:
    spectra: List[Spectrum] = []
    for pdir in find_bruker_pdata_dirs(root_dir):
        try:
            dic, data = ng.bruker.read_pdata(pdir, scale_data=True)
            udic = ng.bruker.guess_udic(dic, data)
            uc = ng.fileiobase.uc_from_udic(udic)
            ppm = uc.ppm_scale()
            sample_name = Path(pdir).parents[1].name
            spectra.append(
                Spectrum(
                    ppm=np.asarray(ppm),
                    intensity=np.asarray(data, dtype=float),
                    sample_name=sample_name,
                    source_path=pdir,
                )
            )
        except Exception:
            continue
    spectra.sort(key=lambda s: natural_sort_key(s.sample_name))
    return spectra


def load_peak_limits_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"Peak identity", "ppm start", "ppm end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[["Peak identity", "ppm start", "ppm end"]].copy()


def estimate_baseline(y: np.ndarray, window_length: int = 51, polyorder: int = 2) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return float(np.median(y))

    wl = min(window_length, y.size if y.size % 2 == 1 else y.size - 1)
    wl = max(wl, 5)
    if wl % 2 == 0:
        wl -= 1

    try:
        smooth = savgol_filter(y, window_length=wl, polyorder=min(polyorder, wl - 2))
        baseline = max(np.min(smooth), np.percentile(y, 5))
        return float(baseline)
    except Exception:
        return float(np.percentile(y, 10))


# -----------------------------------------------------------------------------
# Voigt fitting and integration
# -----------------------------------------------------------------------------

def voigt_profile(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    gamma = max(float(gamma), 1e-8)
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def robust_voigt_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, bool]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 7 or np.allclose(y, y[0]):
        return 0.0, float(x.mean() if x.size else 0.0), 0.01, 0.01, False

    idx = int(np.argmax(y))
    amp0 = float(max(y[idx], 1e-8))
    cen0 = float(x[idx])

    half = amp0 / 2.0
    left, right = idx, idx
    while left > 0 and y[left] > half:
        left -= 1
    while right < x.size - 1 and y[right] > half:
        right += 1

    fwhm = abs(float(x[right] - x[left])) if right > left else max(abs(x[-1] - x[0]) / 10.0, 0.02)
    sigma0 = max(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-3)
    gamma0 = max(fwhm / 2.0, 1e-3)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    width_span = max(abs(x_max - x_min), 0.02)
    lower = [0.0, min(x_min, x_max), 1e-4, 1e-4]
    upper = [amp0 * 5.0 + 1e-6, max(x_min, x_max), width_span, width_span]

    try:
        popt, _ = curve_fit(
            voigt_profile,
            x,
            y,
            p0=[amp0, cen0, sigma0, gamma0],
            bounds=(lower, upper),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]), True
    except Exception:
        return amp0, cen0, sigma0, gamma0, False


def integration_bounds_from_fit(
    ppm: np.ndarray,
    fitted_curve: np.ndarray,
    baseline: float,
    peak_index: int,
    threshold_fraction: float = 0.005,
) -> Tuple[int, int]:
    signal_height = max(float(fitted_curve[peak_index] - baseline), 1e-12)
    threshold = baseline + threshold_fraction * signal_height

    left = peak_index
    while left > 0 and fitted_curve[left] > threshold:
        left -= 1

    right = peak_index
    while right < ppm.size - 1 and fitted_curve[right] > threshold:
        right += 1

    return left, right


def integrate_region(ppm: np.ndarray, signal: np.ndarray, baseline: float, left: int, right: int) -> float:
    x = ppm[left:right + 1]
    y = signal[left:right + 1] - baseline
    return float(np.trapz(y, x=x))


def _orient_signal(y: np.ndarray) -> Tuple[np.ndarray, int]:
    """Fixes spectra that occassionally display upside-down"""
    y = np.asarray(y, dtype=float)
    baseline = np.median(y)
    pos_extent = float(np.max(y) - baseline)
    neg_extent = float(baseline - np.min(y))
    sign = 1 if pos_extent >= neg_extent else -1
    return sign * y, sign


def detect_and_fit_peaks(
    ppm: np.ndarray,
    intensity: np.ndarray,
    sample_name: str,
    height: Optional[float] = None,
    distance: int = 10,
    prominence: Optional[float] = None,
    fit_window: int = 40,
) -> List[PeakResult]:
    y, sign = _orient_signal(intensity)
    baseline = estimate_baseline(y)
    y_corr = y - baseline

    if prominence is None:
        prominence = max(np.std(y_corr) * 0.2, 1e-12)

    peaks, _ = find_peaks(y_corr, height=height, distance=distance, prominence=prominence)
    results: List[PeakResult] = []

    for idx in peaks:
        left = max(0, idx - fit_window)
        right = min(ppm.size - 1, idx + fit_window)

        x_local = ppm[left:right + 1]
        y_local = y_corr[left:right + 1]
        amp, cen, sigma, gamma, success = robust_voigt_fit(x_local, y_local)

        fitted = voigt_profile(ppm, amp, cen, sigma, gamma) + baseline
        int_left, int_right = integration_bounds_from_fit(ppm, fitted, baseline, idx)

        if int_right - int_left < 5:
            int_left = max(0, idx - fit_window)
            int_right = min(ppm.size - 1, idx + fit_window)

        integral = integrate_region(ppm, fitted if success else y, baseline, int_left, int_right)

        results.append(
            PeakResult(
                sample_name=sample_name,
                peak_name=None,
                index=int(idx),
                ppm_center=float(cen),
                amplitude=float(amp * sign),
                sigma=float(sigma),
                gamma=float(gamma),
                integral=float(integral),
                start_ppm=float(ppm[int_left]),
                end_ppm=float(ppm[int_right]),
                baseline=float(baseline * sign),
                fit_success=bool(success),
            )
        )

    return results


def integrate_known_peaks(
    ppm: np.ndarray,
    intensity: np.ndarray,
    sample_name: str,
    peak_limits: pd.DataFrame,
) -> List[PeakResult]:
    y, sign = _orient_signal(intensity)
    baseline = estimate_baseline(y)
    y_corr = y - baseline
    results: List[PeakResult] = []

    for _, row in peak_limits.iterrows():
        peak_name = str(row["Peak identity"])
        start_ppm = float(row["ppm start"])
        end_ppm = float(row["ppm end"])

        s = int(np.argmin(np.abs(ppm - start_ppm)))
        e = int(np.argmin(np.abs(ppm - end_ppm)))
        if s > e:
            s, e = e, s

        x_region = ppm[s:e + 1]
        y_region = y_corr[s:e + 1]
        idx_local = int(np.argmax(y_region)) if y_region.size else 0
        idx_global = s + idx_local

        if x_region.size >= 7:
            amp, cen, sigma, gamma, success = robust_voigt_fit(x_region, y_region)
            fitted = voigt_profile(ppm, amp, cen, sigma, gamma) + baseline
            integral = integrate_region(ppm, fitted if success else y, baseline, s, e)
        else:
            amp, cen, sigma, gamma, success = 0.0, float(np.mean(x_region)), 0.0, 0.0, False
            integral = integrate_region(ppm, y, baseline, s, e)

        results.append(
            PeakResult(
                sample_name=sample_name,
                peak_name=peak_name,
                index=int(idx_global),
                ppm_center=float(cen),
                amplitude=float(amp * sign),
                sigma=float(sigma),
                gamma=float(gamma),
                integral=float(integral),
                start_ppm=float(ppm[s]),
                end_ppm=float(ppm[e]),
                baseline=float(baseline * sign),
                fit_success=bool(success),
            )
        )

    return results


# -----------------------------------------------------------------------------
# Alignment and statistics
# -----------------------------------------------------------------------------

def align_peaks_across_samples(
    peaks_by_sample: List[List[PeakResult]],
    eps: float = 0.02,
    min_samples: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    ppm_values: List[float] = []
    owners: List[Tuple[int, int]] = []

    for sample_idx, sample_peaks in enumerate(peaks_by_sample):
        for peak_idx, peak in enumerate(sample_peaks):
            ppm_values.append(peak.ppm_center)
            owners.append((sample_idx, peak_idx))

    if not ppm_values:
        return pd.DataFrame(), []

    X = np.array(ppm_values).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    unique_clusters = sorted([c for c in set(labels) if c != -1], key=lambda c: X[labels == c].mean())
    cluster_to_name = {c: f"Peak_{i + 1}" for i, c in enumerate(unique_clusters)}

    rows: List[Dict[str, float]] = []
    sample_names = []
    columns = list(cluster_to_name.values())

    for sample_idx, sample_peaks in enumerate(peaks_by_sample):
        sample_name = sample_peaks[0].sample_name if sample_peaks else f"Sample_{sample_idx + 1}"
        sample_names.append(sample_name)
        row: Dict[str, float] = {"Sample": sample_name}
        for col in columns:
            row[col] = 0.0
        rows.append(row)

    for label, (sample_idx, peak_idx) in zip(labels, owners):
        if label == -1:
            continue
        col = cluster_to_name[label]
        rows[sample_idx][col] = peaks_by_sample[sample_idx][peak_idx].integral

    return pd.DataFrame(rows), columns


def run_pca(df: pd.DataFrame, n_components: int = 2, scale: bool = True):
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if scale:
        X = StandardScaler().fit_transform(X)
    model = PCA(n_components=n_components)
    scores = model.fit_transform(X)
    return model, scores


def run_plsda(df: pd.DataFrame, labels: List[str], n_components: int = 2, scale: bool = True):
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    y = np.asarray(labels)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of labels must match number of samples.")
    if scale:
        X = StandardScaler().fit_transform(X)
    y_encoded = LabelEncoder().fit_transform(y)
    model = PLSRegression(n_components=n_components)
    scores = model.fit_transform(X, y_encoded)[0]
    return model, scores


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

class NMRWorkbench:
    def __init__(self) -> None:
        Theme.apply_matplotlib()

        self.root = tk.Tk()
        self.root.title("NMR Voigt Integration Workbench")
        self.root.geometry("1500x920")
        self.root.configure(bg=Theme.BG)

        self._setup_ttk_style()

        self.spectra: List[Spectrum] = []
        self.peak_limits = pd.DataFrame()
        self.detected_peaks: List[List[PeakResult]] = []
        self.known_peaks: List[List[PeakResult]] = []
        self.peak_matrix: Optional[pd.DataFrame] = None
        self.peak_matrix_columns: List[str] = []
        self.current_index = 0

        self._build_ui()
        self._refresh_status()

    def _setup_ttk_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=Theme.BG)
        style.configure("TLabelframe", background=Theme.BG, foreground=Theme.FG)
        style.configure("TLabelframe.Label", background=Theme.BG, foreground=Theme.FG)
        style.configure("TLabel", background=Theme.BG, foreground=Theme.FG)
        style.configure("TButton", background=Theme.PANEL, foreground=Theme.FG)
        style.configure("TCheckbutton", background=Theme.BG, foreground=Theme.FG)
        style.configure("TRadiobutton", background=Theme.BG, foreground=Theme.FG)
        style.configure("TEntry", fieldbackground="white")

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=(12, 6))
        ttk.Label(header, text="NMR Voigt Integration Workbench", font=("Segoe UI", 20, "bold")).pack(side="left")
        self.header_status = ttk.Label(header, text="Ready")
        self.header_status.pack(side="right")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        left_wrap = ttk.Frame(body)
        left_wrap.pack(side="left", fill="y")

        scroll_canvas = tk.Canvas(left_wrap, bg=Theme.BG, width=335, highlightthickness=0)
        scroll_canvas.pack(side="left", fill="y", expand=False)
        ybar = ttk.Scrollbar(left_wrap, orient="vertical", command=scroll_canvas.yview)
        ybar.pack(side="right", fill="y")
        scroll_canvas.configure(yscrollcommand=ybar.set)

        self.sidebar = ttk.Frame(scroll_canvas)
        scroll_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        self.sidebar.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
        )

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True, padx=(16, 0))

        self.figure = plt.Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

        self._build_sidebar()

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken").pack(side="bottom", fill="x")

    def _build_sidebar(self) -> None:
        frame = ttk.LabelFrame(self.sidebar, text="Data")
        frame.pack(fill="x", padx=6, pady=6)
        ttk.Button(frame, text="Load Spectra", command=self.load_spectra).pack(fill="x", padx=6, pady=3)
        self.spectra_label = ttk.Label(frame, text="No spectra loaded")
        self.spectra_label.pack(anchor="w", padx=6, pady=2)
        ttk.Button(frame, text="Load Peak Limits", command=self.load_peak_limits).pack(fill="x", padx=6, pady=3)
        self.peak_limits_label = ttk.Label(frame, text="No peak limits file")
        self.peak_limits_label.pack(anchor="w", padx=6, pady=2)

        frame = ttk.LabelFrame(self.sidebar, text="Spectra")
        frame.pack(fill="x", padx=6, pady=6)
        self.spectrum_list = tk.Listbox(
            frame,
            height=8,
            bg=Theme.PANEL,
            fg=Theme.FG,
            selectbackground="#4c78a8",
        )
        self.spectrum_list.pack(fill="x", padx=6, pady=6)
        self.spectrum_list.bind("<<ListboxSelect>>", self._on_spectrum_selected)

        frame = ttk.LabelFrame(self.sidebar, text="Detection / Alignment Parameters")
        frame.pack(fill="x", padx=6, pady=6)
        self.height_entry = self._labeled_entry(frame, "Height threshold", "")
        self.distance_entry = self._labeled_entry(frame, "Distance", "10")
        self.prominence_entry = self._labeled_entry(frame, "Prominence factor", "0.2")
        self.eps_entry = self._labeled_entry(frame, "Alignment tolerance (ppm)", "0.02")

        frame = ttk.LabelFrame(self.sidebar, text="Actions")
        frame.pack(fill="x", padx=6, pady=6)
        ttk.Button(frame, text="Detect Peaks (Current)", command=self.detect_current).pack(fill="x", padx=6, pady=3)
        ttk.Button(frame, text="Detect Peaks (All)", command=self.detect_all).pack(fill="x", padx=6, pady=3)
        ttk.Button(frame, text="Integrate Known (Current)", command=self.integrate_known_current).pack(fill="x", padx=6, pady=3)
        ttk.Button(frame, text="Integrate Known (All)", command=self.integrate_known_all).pack(fill="x", padx=6, pady=3)
        ttk.Button(frame, text="Align Detected Peaks", command=self.align_detected).pack(fill="x", padx=6, pady=3)

        frame = ttk.LabelFrame(self.sidebar, text="Display")
        frame.pack(fill="x", padx=6, pady=6)
        self.show_detected_var = tk.BooleanVar(value=True)
        self.show_known_var = tk.BooleanVar(value=True)
        self.show_baseline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Show detected peaks", variable=self.show_detected_var, command=self.plot_current).pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(frame, text="Show known peaks", variable=self.show_known_var, command=self.plot_current).pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(frame, text="Show baseline", variable=self.show_baseline_var, command=self.plot_current).pack(anchor="w", padx=6, pady=2)

        frame = ttk.LabelFrame(self.sidebar, text="Statistics")
        frame.pack(fill="x", padx=6, pady=6)
        self.stats_method = tk.StringVar(value="PCA")
        ttk.Radiobutton(frame, text="PCA", value="PCA", variable=self.stats_method).pack(anchor="w", padx=6, pady=2)
        ttk.Radiobutton(frame, text="PLS-DA", value="PLS-DA", variable=self.stats_method).pack(anchor="w", padx=6, pady=2)
        self.ncomp_spin = ttk.Spinbox(frame, from_=2, to=10, width=6)
        self.ncomp_spin.set(2)
        ttk.Label(frame, text="Components").pack(anchor="w", padx=6, pady=(6, 0))
        self.ncomp_spin.pack(anchor="w", padx=6, pady=(0, 4))
        self.scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Scale features", variable=self.scale_var).pack(anchor="w", padx=6, pady=2)
        ttk.Label(frame, text="PLS-DA labels (comma-separated)").pack(anchor="w", padx=6, pady=(6, 0))
        self.labels_entry = ttk.Entry(frame)
        self.labels_entry.pack(fill="x", padx=6, pady=4)
        ttk.Button(frame, text="Run Statistics", command=self.run_statistics).pack(fill="x", padx=6, pady=4)

        frame = ttk.LabelFrame(self.sidebar, text="Export")
        frame.pack(fill="x", padx=6, pady=6)
        ttk.Button(frame, text="Export Excel Report", command=self.export_excel_report).pack(fill="x", padx=6, pady=4)

    def _labeled_entry(self, parent: ttk.Frame, label: str, default: str) -> ttk.Entry:
        ttk.Label(parent, text=label).pack(anchor="w", padx=6, pady=(4, 0))
        entry = ttk.Entry(parent)
        entry.insert(0, default)
        entry.pack(fill="x", padx=6, pady=(0, 4))
        return entry

    def _set_busy(self, message: str) -> None:
        self.header_status.config(text=message)
        self.status_var.set(message)
        self.root.update_idletasks()

    def _set_ready(self, message: str = "Ready") -> None:
        self.header_status.config(text=message)
        self.status_var.set(message)
        self.root.update_idletasks()

    def _refresh_status(self) -> None:
        detected = sum(len(x) for x in self.detected_peaks)
        known = sum(len(x) for x in self.known_peaks)
        self.status_var.set(
            f"Spectra: {len(self.spectra)} | Detected peaks: {detected} | Known peaks: {known}"
        )

    def _on_spectrum_selected(self, event=None) -> None:
        sel = self.spectrum_list.curselection()
        if sel:
            self.current_index = int(sel[0])
            self.plot_current()

    def _current_spectrum(self) -> Optional[Spectrum]:
        if not self.spectra:
            return None
        return self.spectra[self.current_index]

    def _parse_detection_parameters(self) -> Tuple[Optional[float], int, float]:
        height_text = self.height_entry.get().strip()
        height = float(height_text) if height_text else None
        distance = int(self.distance_entry.get().strip())
        prom_factor = float(self.prominence_entry.get().strip())
        return height, distance, prom_factor

    def load_spectra(self) -> None:
        use_zip = messagebox.askyesno("Load spectra", "Is your data in a ZIP file?")
        if use_zip:
            path = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
            if not path:
                return
            root_dir = extract_zip(path)
        else:
            root_dir = filedialog.askdirectory(title="Select root directory containing Bruker data")
            if not root_dir:
                return

        self._set_busy("Loading spectra...")
        spectra = load_bruker_spectra(root_dir)
        if not spectra:
            self._set_ready()
            messagebox.showerror("Load error", "No Bruker processed spectra were found.")
            return

        self.spectra = spectra
        self.detected_peaks = [[] for _ in spectra]
        self.known_peaks = [[] for _ in spectra]
        self.current_index = 0

        self.spectrum_list.delete(0, tk.END)
        for spec in spectra:
            self.spectrum_list.insert(tk.END, spec.sample_name)
        self.spectrum_list.select_set(0)

        self.spectra_label.config(text=f"{len(spectra)} spectra loaded")
        self.plot_current()
        self._set_ready()
        self._refresh_status()

    def load_peak_limits(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if not path:
            return
        try:
            self.peak_limits = load_peak_limits_excel(path)
            self.peak_limits_label.config(text=os.path.basename(path))
        except Exception as exc:
            messagebox.showerror("Peak limits error", str(exc))
            return
        self._refresh_status()

    def detect_current(self) -> None:
        spec = self._current_spectrum()
        if spec is None:
            messagebox.showerror("Error", "Load spectra first.")
            return
        try:
            height, distance, prom_factor = self._parse_detection_parameters()
        except Exception:
            messagebox.showerror("Error", "Invalid detection parameters.")
            return

        prominence = prom_factor * np.std(spec.intensity) if prom_factor > 0 else None
        peaks = detect_and_fit_peaks(
            spec.ppm,
            spec.intensity,
            spec.sample_name,
            height=height,
            distance=distance,
            prominence=prominence,
        )
        self.detected_peaks[self.current_index] = peaks
        self.plot_current()
        self._refresh_status()

    def detect_all(self) -> None:
        if not self.spectra:
            messagebox.showerror("Error", "Load spectra first.")
            return
        try:
            height, distance, prom_factor = self._parse_detection_parameters()
        except Exception:
            messagebox.showerror("Error", "Invalid detection parameters.")
            return

        self._set_busy("Detecting peaks in all spectra...")
        for i, spec in enumerate(self.spectra):
            prominence = prom_factor * np.std(spec.intensity) if prom_factor > 0 else None
            self.detected_peaks[i] = detect_and_fit_peaks(
                spec.ppm,
                spec.intensity,
                spec.sample_name,
                height=height,
                distance=distance,
                prominence=prominence,
            )
        self.plot_current()
        self._set_ready("Peak detection completed")
        self._refresh_status()
        messagebox.showinfo("Done", f"Detected peaks in {len(self.spectra)} spectra.")

    def integrate_known_current(self) -> None:
        spec = self._current_spectrum()
        if spec is None:
            messagebox.showerror("Error", "Load spectra first.")
            return
        if self.peak_limits.empty:
            messagebox.showerror("Error", "Load a peak limits file first.")
            return

        self.known_peaks[self.current_index] = integrate_known_peaks(
            spec.ppm,
            spec.intensity,
            spec.sample_name,
            self.peak_limits,
        )
        self.plot_current()
        self._refresh_status()

    def integrate_known_all(self) -> None:
        if not self.spectra:
            messagebox.showerror("Error", "Load spectra first.")
            return
        if self.peak_limits.empty:
            messagebox.showerror("Error", "Load a peak limits file first.")
            return

        self._set_busy("Integrating known regions in all spectra...")
        for i, spec in enumerate(self.spectra):
            self.known_peaks[i] = integrate_known_peaks(
                spec.ppm,
                spec.intensity,
                spec.sample_name,
                self.peak_limits,
            )
        self.plot_current()
        self._set_ready("Known-peak integration completed")
        self._refresh_status()
        messagebox.showinfo("Done", f"Integrated known peaks in {len(self.spectra)} spectra.")

    def align_detected(self) -> None:
        if not any(self.detected_peaks):
            messagebox.showerror("Error", "Run peak detection first.")
            return
        try:
            eps = float(self.eps_entry.get().strip())
        except Exception:
            eps = 0.02

        matrix, columns = align_peaks_across_samples(self.detected_peaks, eps=eps)
        self.peak_matrix = matrix
        self.peak_matrix_columns = columns
        messagebox.showinfo("Alignment complete", f"Created matrix with {len(columns)} aligned peak columns.")

    def plot_current(self) -> None:
        spec = self._current_spectrum()
        if spec is None:
            return

        ppm = spec.ppm
        intensity = spec.intensity
        baseline = estimate_baseline(intensity)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(ppm, intensity, linewidth=1.0, label="Spectrum")

        if self.show_baseline_var.get():
            ax.axhline(baseline, color=Theme.BASELINE, linestyle="--", linewidth=1.0, label="Baseline")

        if self.show_detected_var.get() and self.current_index < len(self.detected_peaks):
            for peak in self.detected_peaks[self.current_index]:
                curve = voigt_profile(ppm, peak.amplitude, peak.ppm_center, peak.sigma, peak.gamma) + baseline
                s = int(np.argmin(np.abs(ppm - peak.start_ppm)))
                e = int(np.argmin(np.abs(ppm - peak.end_ppm)))
                if s > e:
                    s, e = e, s
                ax.fill_between(ppm[s:e + 1], baseline, curve[s:e + 1], alpha=0.28, color=Theme.DETECT_FILL)
                ax.plot(ppm, curve, linestyle="--", linewidth=0.8, alpha=0.8)
                y_marker = intensity[peak.index]
                ax.plot(peak.ppm_center, y_marker, marker="x", color=Theme.PEAK)
                ax.text(
                    peak.ppm_center,
                    y_marker * 1.03,
                    f"{peak.integral:.2e}",
                    fontsize=7,
                    ha="center",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.45),
                )

        if self.show_known_var.get() and self.current_index < len(self.known_peaks):
            for peak in self.known_peaks[self.current_index]:
                s = int(np.argmin(np.abs(ppm - peak.start_ppm)))
                e = int(np.argmin(np.abs(ppm - peak.end_ppm)))
                if s > e:
                    s, e = e, s
                if peak.fit_success:
                    curve = voigt_profile(ppm, peak.amplitude, peak.ppm_center, peak.sigma, peak.gamma) + baseline
                else:
                    curve = np.full_like(ppm, baseline)
                ax.fill_between(ppm[s:e + 1], baseline, curve[s:e + 1], alpha=0.22, color=Theme.KNOWN_FILL)
                ax.plot(ppm, curve, linestyle=":", linewidth=1.0)
                ymax = float(np.max(intensity[s:e + 1])) if e >= s else float(intensity[s])
                label = peak.peak_name if peak.peak_name else "Known"
                ax.text(
                    peak.ppm_center,
                    ymax * 1.05,
                    f"{label}\\n{peak.integral:.2e}",
                    fontsize=7,
                    ha="center",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.45),
                )

        ax.set_title(f"{spec.sample_name}")
        ax.set_xlabel("ppm")
        ax.set_ylabel("Intensity")
        ax.invert_xaxis()
        ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    def run_statistics(self) -> None:
        if self.peak_matrix is None or self.peak_matrix.empty:
            messagebox.showerror("Error", "Create an aligned peak matrix first.")
            return

        df = self.peak_matrix.copy()
        X = df.drop(columns=["Sample"], errors="ignore")
        if X.empty:
            messagebox.showerror("Error", "No numeric peak matrix available.")
            return

        n_components = int(self.ncomp_spin.get())
        n_components = max(2, min(n_components, X.shape[1], X.shape[0]))
        scale = bool(self.scale_var.get())
        method = self.stats_method.get()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if method == "PCA":
            model, scores = run_pca(X, n_components=n_components, scale=scale)
            ax.scatter(scores[:, 0], scores[:, 1])
            for i, name in enumerate(df["Sample"]):
                ax.annotate(str(name), (scores[i, 0], scores[i, 1]), fontsize=8)
            evr = model.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
            ax.set_title("PCA Scores")
        else:
            label_text = self.labels_entry.get().strip()
            if not label_text:
                messagebox.showerror("Error", "Enter comma-separated class labels for PLS-DA.")
                return
            labels = [x.strip() for x in label_text.split(",") if x.strip()]
            if len(labels) != len(df):
                messagebox.showerror("Error", f"Expected {len(df)} labels, got {len(labels)}.")
                return
            _, scores = run_plsda(X, labels, n_components=n_components, scale=scale)
            unique_labels = sorted(set(labels))
            cmap = plt.cm.get_cmap("tab10", len(unique_labels))
            for i, group in enumerate(unique_labels):
                mask = [j for j, v in enumerate(labels) if v == group]
                ax.scatter(scores[mask, 0], scores[mask, 1], label=group, color=cmap(i))
                for j in mask:
                    ax.annotate(str(df.iloc[j]["Sample"]), (scores[j, 0], scores[j, 1]), fontsize=8)
            ax.legend()
            ax.set_xlabel("LV1")
            ax.set_ylabel("LV2")
            ax.set_title("PLS-DA Scores")

        self.canvas.draw()

    def _peaks_to_dataframe(self, peaks_by_sample: List[List[PeakResult]], include_name: bool = True) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for sample_peaks in peaks_by_sample:
            for peak in sample_peaks:
                row = asdict(peak)
                row["fwhm_approx"] = peak.fwhm_approx
                if not include_name:
                    row.pop("peak_name", None)
                rows.append(row)
        return pd.DataFrame(rows)

    def export_excel_report(self) -> None:
        if not self.spectra:
            messagebox.showerror("Error", "Nothing to export. Load spectra first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if not path:
            return

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                overview = pd.DataFrame(
                    {
                        "Parameter": [
                            "Spectra loaded",
                            "Peak limits loaded",
                            "Detected peak count",
                            "Known peak count",
                        ],
                        "Value": [
                            len(self.spectra),
                            not self.peak_limits.empty,
                            sum(len(x) for x in self.detected_peaks),
                            sum(len(x) for x in self.known_peaks),
                        ],
                    }
                )
                overview.to_excel(writer, sheet_name="Overview", index=False)

                pd.DataFrame(
                    [{"Sample": s.sample_name, "Source": s.source_path} for s in self.spectra]
                ).to_excel(writer, sheet_name="Samples", index=False)

                if not self.peak_limits.empty:
                    self.peak_limits.to_excel(writer, sheet_name="Peak Limits", index=False)

                detected_df = self._peaks_to_dataframe(self.detected_peaks, include_name=False)
                if not detected_df.empty:
                    detected_df.to_excel(writer, sheet_name="Detected Peaks", index=False)

                known_df = self._peaks_to_dataframe(self.known_peaks, include_name=True)
                if not known_df.empty:
                    known_df.to_excel(writer, sheet_name="Known Peaks", index=False)

                if self.peak_matrix is not None and not self.peak_matrix.empty:
                    self.peak_matrix.to_excel(writer, sheet_name="Peak Matrix", index=False)

            messagebox.showinfo("Export complete", f"Saved report to:{path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = NMRWorkbench()
    app.run()