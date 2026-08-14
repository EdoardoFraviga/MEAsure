from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import gaussian_kde, gmean, kurtosis, skew

try:
    from .propagation import run_propagation_analysis, write_propagation_summary
except ImportError:
    from propagation import run_propagation_analysis, write_propagation_summary


Log = Callable[[str], None]


@dataclass
class AnalysisConfig:
    inputs: list[str]
    output_dir: str
    recursive: bool = True
    well: str = "Well_A1"
    active_threshold_lfp: float = 0.8
    active_threshold_mua: float = 0.0
    sne_groups: list[str] = field(default_factory=lambda: ["All groups"])
    sne_min_active_percent: float = 40.0
    kde_bandwidth: float = 0.01
    boundary_percent: float = 30.0
    fano_window_s: float = 10.0
    acf_max_lag: int = 100
    raster_start_s: float = 0.0
    raster_end_s: float = 60.0
    generate_raster: bool = True
    aggregations: set[str] = field(default_factory=lambda: {"mean", "geometric_mean"})
    lfp_metrics: set[str] = field(default_factory=lambda: {
        "frequency_distribution_stats", "frequency", "active_electrodes", "amplitude",
        "cv_iei", "lfp_sne_ratio", "lfp_sne_kdl_ratio", "electrode_frequencies",
    })
    mua_metrics: set[str] = field(default_factory=lambda: {
        "frequency_distribution_stats", "frequency", "cv_iei", "active_electrodes",
        "amplitude", "mua_lfp_ratio", "electrode_frequencies",
    })
    sne_metrics: set[str] = field(default_factory=lambda: {
        "sne_frequency", "sne_kdl_frequency", "sne_duration", "cv_iei",
        "fano", "psd_peaks", "acf_peaks",
    })
    plots: set[str] = field(default_factory=lambda: {
        "sync", "sync_kde", "frequency_distribution_comparison",
    })
    raster_formats: set[str] = field(default_factory=lambda: {"png", "svg", "emf.svg"})
    propagation_enabled: bool = False
    propagation_regions: set[str] = field(default_factory=lambda: {"HP", "CTX"})
    propagation_metrics: set[str] = field(default_factory=lambda: {
        "duration", "total_distance", "maximum_distance", "total_speed",
        "expansion_distance", "expansion_speed", "retraction_distance", "retraction_speed",
    })
    route_metrics: set[str] = field(default_factory=lambda: {
        "average_occurrence", "overall_occurrence", "full_routes",
        "sub_routes", "signal_origin", "signal_end",
    })
    propagation_plots: set[str] = field(default_factory=lambda: {"lfp_heatmap", "cats", "weighted_cats"})
    propagation_min_active_percent: float = 40.0
    propagation_boundary_percent: float = 0.0
    propagation_bins: int = 200
    centroid_step_s: float = 0.01
    centroid_tolerance_s: float = 0.025
    electrode_pitch_um: float = 81.0
    route_group_names: list[str] = field(default_factory=lambda: ["CA1", "CA2", "CA3", "DG", "SUBV"])
    network_activity_enabled: bool = True
    analyze_lfp: bool = True
    analyze_mua: bool = True


def load_config(path: str | Path) -> AnalysisConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "aggregations", "lfp_metrics", "mua_metrics", "sne_metrics", "plots", "raster_formats",
        "propagation_regions", "propagation_metrics", "route_metrics", "propagation_plots",
    ):
        raw[key] = set(raw.get(key, []))
    return AnalysisConfig(**raw)


def save_config(config: AnalysisConfig, path: str | Path) -> None:
    raw = vars(config).copy()
    for key in (
        "aggregations", "lfp_metrics", "mua_metrics", "sne_metrics", "plots", "raster_formats",
        "propagation_regions", "propagation_metrics", "route_metrics", "propagation_plots",
    ):
        raw[key] = sorted(raw[key])
    Path(path).write_text(json.dumps(raw, indent=2), encoding="utf-8")


def discover_recordings(inputs: list[str], recursive: bool) -> list[Path]:
    found: set[Path] = set()
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_file() and path.suffix.lower() in {".brx", ".bxr"}:
            found.add(path.resolve())
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            found.update(p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in {".brx", ".bxr"})
    return sorted(found)


def load_groups(recording: Path) -> tuple[list[str], list[np.ndarray]]:
    json_path = recording.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Matching group file not found: {json_path.name}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    groups = payload.get("Groups", [])
    names = [str(g["UserDefinedName"]) for g in groups]
    indexes = [np.asarray(g["PixelIndexes"], dtype=int) for g in groups]
    if not names:
        raise ValueError("The JSON file contains no electrode groups.")
    return names, indexes


def _dataset(group: h5py.Group, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in group:
            return np.asarray(group[name]).reshape(-1)
    return None


def _read_streams(path: Path, well: str) -> tuple[float, float, dict[str, tuple[np.ndarray, np.ndarray]], h5py.File]:
    h5 = h5py.File(path, "r")
    if well not in h5:
        h5.close()
        raise KeyError(f"Well '{well}' is not present.")
    sr = float(np.asarray(h5.attrs.get("SamplingRate", 1.0)).reshape(-1)[0])
    group = h5[well]
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for signal, ch_names, time_names in (
        ("LFP", ("FpChIdxs",), ("FpTimes",)),
        ("MUA", ("SpikeChIdxs",), ("SpikeTimes",)),
    ):
        channels = _dataset(group, ch_names)
        times = _dataset(group, time_names)
        if channels is not None and times is not None:
            n = min(len(channels), len(times))
            streams[signal] = (channels[:n].astype(int), times[:n].astype(float) / sr)
    duration = 0.0
    if "TOC" in h5:
        toc = np.asarray(h5["TOC"])
        if toc.size:
            duration = float(toc.reshape(-1, toc.shape[-1])[-1, -1]) / sr
    if duration <= 0 and streams:
        duration = max((float(t.max()) if len(t) else 0.0) for _, t in streams.values())
    return sr, duration, streams, h5


def _event_amplitudes(h5: h5py.File, well: str, signal: str, channels: np.ndarray) -> dict[int, float]:
    group = h5[well]
    prefix = "Fp" if signal == "LFP" else "Spike"
    forms_name = prefix + "Forms"
    idx_name = prefix + "ChIdxs"
    if forms_name not in group or idx_name not in group:
        return {}
    forms = np.asarray(group[forms_name]).reshape(-1)
    event_channels = np.asarray(group[idx_name]).reshape(-1).astype(int)
    wavelength = group[forms_name].attrs.get("Wavelength", group.attrs.get("Wavelength", 0))
    wavelength = int(np.asarray(wavelength).reshape(-1)[0]) if np.asarray(wavelength).size else 0
    if wavelength <= 0 or len(forms) < wavelength:
        return {}
    attrs = h5.attrs
    try:
        max_a = float(np.asarray(attrs["MaxAnalogValue"]).reshape(-1)[0])
        min_a = float(np.asarray(attrs["MinAnalogValue"]).reshape(-1)[0])
        max_d = float(np.asarray(attrs["MaxDigitalValue"]).reshape(-1)[0])
        min_d = float(np.asarray(attrs["MinDigitalValue"]).reshape(-1)[0])
        scale = (max_a - min_a) / (max_d - min_d)
    except (KeyError, ZeroDivisionError):
        scale = 1.0
    result: dict[int, float] = {}
    for channel in np.unique(channels):
        hits = np.flatnonzero(event_channels == channel)
        values = []
        for hit in hits:
            wave = forms[hit * wavelength:(hit + 1) * wavelength]
            if len(wave):
                values.append(float(np.ptp(wave)) * scale)
        if values:
            result[int(channel)] = float(np.nanmean(values))
    return result


def _regularity(event_times: np.ndarray, fano_window: float) -> dict[str, object]:
    event_times = np.sort(np.asarray(event_times, dtype=float))
    iei = np.diff(event_times)
    cv = float(np.std(iei) / np.mean(iei)) if len(iei) and np.mean(iei) else np.nan
    if len(iei) and np.all(iei > 0):
        gcv = float(np.sqrt(np.exp(np.std(np.log(iei)) ** 2) - 1))
    else:
        gcv = np.nan
    if len(event_times) > 1 and event_times[-1] - event_times[0] >= fano_window:
        edges = np.arange(event_times[0], event_times[-1] + fano_window, fano_window)
        counts, _ = np.histogram(event_times, edges)
        fano = float(np.var(counts) / np.mean(counts)) if np.mean(counts) else np.nan
    else:
        fano = np.nan
    return {"iei": iei, "cv_iei": cv, "gcv_iei": gcv, "fano": fano}


def _detect_snes(times: np.ndarray, channels: np.ndarray, electrodes: np.ndarray, config: AnalysisConfig):
    mask = np.isin(channels, electrodes)
    t = times[mask]
    c = channels[mask]
    if len(t) < 2 or np.unique(t).size < 2:
        return np.array([]), [], None
    kde = gaussian_kde(t, bw_method=config.kde_bandwidth)
    grid = np.linspace(float(t.min()), float(t.max()), min(5000, max(1000, len(t) * 2)))
    density = kde(grid)
    peaks, _ = find_peaks(density, prominence=max(density.max() * 0.002, np.finfo(float).eps))
    threshold = density.max() * config.boundary_percent / 100.0
    events, durations, valid_peaks = [], [], []
    minimum = max(1, math.ceil(len(np.unique(electrodes)) * config.sne_min_active_percent / 100.0))
    for peak in peaks:
        if density[peak] < threshold:
            continue
        left = peak
        right = peak
        while left > 0 and density[left] >= threshold:
            left -= 1
        while right < len(grid) - 1 and density[right] >= threshold:
            right += 1
        event_mask = (t >= grid[left]) & (t <= grid[right])
        if np.unique(c[event_mask]).size >= minimum:
            events.append(float(np.mean(t[event_mask])))
            durations.append(float(grid[right] - grid[left]))
            valid_peaks.append(peak)
    return np.asarray(events), durations, (grid, density, np.asarray(valid_peaks), threshold)


def _detect_snes_histogram(
    times: np.ndarray, channels: np.ndarray, electrodes: np.ndarray,
    config: AnalysisConfig, bins: int = 200,
):
    mask = np.isin(channels, electrodes)
    t = times[mask]
    c = channels[mask]
    if len(t) < 2:
        return np.array([]), None
    hist, edges = np.histogram(t, bins=bins)
    minimum = max(1, math.ceil(len(np.unique(electrodes)) * config.sne_min_active_percent / 100.0))
    peaks, _ = find_peaks(hist, height=minimum)
    events, valid_peaks = [], []
    for peak in peaks:
        event_mask = (t >= edges[peak]) & (t < edges[peak + 1])
        if np.unique(c[event_mask]).size >= minimum:
            events.append(float((edges[peak] + edges[peak + 1]) / 2))
            valid_peaks.append(peak)
    return np.asarray(events), (hist, edges, np.asarray(valid_peaks), minimum)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _save_raster(times, channels, output_base: Path, start: float, end: float, formats: set[str]):
    mask = (times >= start) & (times <= end)
    unique = np.unique(channels[mask])
    data = [times[mask & (channels == ch)] for ch in unique]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.eventplot(data, colors="black", linelengths=.7, linewidths=.5)
    ax.set(xlim=(start, end), title=f"Raster plot ({start:g}–{end:g} s)", xlabel="Time (s)", ylabel="Electrode")
    fig.tight_layout()
    for fmt in formats:
        suffix = ".emf.svg" if fmt == "emf.svg" else f".{fmt}"
        fig.savefig(str(output_base) + suffix, format="svg" if fmt == "emf.svg" else fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_distribution(values: np.ndarray, output: Path, kind: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    if kind == "violin":
        if len(values):
            ax.violinplot(values, showmeans=True)
        ax.set_xticks([1], [title])
    else:
        ax.hist(values, bins="auto", density=True, alpha=.7, edgecolor="black")
    ax.set(title=title, ylabel="Density" if kind == "hist" else "Frequency", xlabel="Frequency")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _save_frequency_distribution_comparison(values: np.ndarray, output: Path, title: str):
    from scipy.stats import lognorm, norm

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(values):
        ax.hist(values, bins="auto", density=True, alpha=.5, label="Observed", edgecolor="black")
    if len(values) > 1 and np.std(values) > 0:
        x_start = max(0.0, float(values.min()))
        x = np.linspace(x_start, float(values.max()), 500)
        mu, std = norm.fit(values)
        ax.plot(x, norm.pdf(x, mu, std), linewidth=2, label=f"Normal (μ={mu:.2f}, σ={std:.2f})")
        positive = values[values > 0]
        if len(positive) > 1:
            shape, loc, scale = lognorm.fit(positive, floc=0)
            ax.plot(x, lognorm.pdf(x, shape, loc=loc, scale=scale), linewidth=2, label="Lognormal")
    ax.set(title=f"Firing Rate Distribution — {title}", xlabel="Frequency", ylabel="Density")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _save_psd(iei: np.ndarray, output: Path) -> object:
    if len(iei) < 2:
        return np.nan
    freq, power = welch(iei, fs=1.0)
    peaks, _ = find_peaks(power, prominence=max(float(power.max()) * .01, np.finfo(float).eps))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freq, power)
    ax.scatter(freq[peaks], power[peaks], color="red", s=18)
    ax.set(title="SNE IEI power spectral density", xlabel="Frequency (Hz)", ylabel="Power")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return freq[peaks].tolist()


def _save_acf(iei: np.ndarray, output: Path, max_lag: int) -> object:
    if len(iei) < 2:
        return np.nan
    centered = iei - np.mean(iei)
    acf = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    if acf[0]:
        acf = acf / acf[0]
    acf = acf[:max_lag]
    peaks, _ = find_peaks(acf, prominence=.1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(acf)
    ax.scatter(peaks, acf[peaks], color="red", s=18)
    ax.set(title="SNE IEI autocorrelation", xlabel="Lag", ylabel="Autocorrelation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return peaks.tolist()


def _aggregate(values: np.ndarray | list[float], aggregations: set[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result: dict[str, float] = {}
    if "mean" in aggregations:
        result["Mean"] = float(np.mean(values)) if len(values) else np.nan
    if "geometric_mean" in aggregations:
        positive = values[values > 0]
        result["Geometric_mean"] = float(gmean(positive)) if len(positive) else np.nan
    if "median" in aggregations:
        result["Median"] = float(np.median(values)) if len(values) else np.nan
    return result


def _per_electrode_cv(times: np.ndarray, channels: np.ndarray) -> list[float]:
    values = []
    for channel in np.unique(channels):
        iei = np.diff(np.sort(times[channels == channel]))
        if len(iei) and np.mean(iei):
            values.append(float(np.std(iei) / np.mean(iei)))
    return values


def _write_signal_summary(rows: list[dict], path: Path) -> None:
    """Write compact, signal-specific worksheets with readable scientific tables."""
    frame = pd.DataFrame(rows)
    preferred = {
        "LFP": [
            "File", "Group", "Duration_s", "Sampling_rate_Hz",
            "Rate_skewness", "Rate_kurtosis", "Rate_gini",
        ],
        "SNE": ["File", "Group"],
        "MUA": [
            "File", "Group", "Duration_s", "Sampling_rate_Hz",
            "Rate_skewness", "Rate_kurtosis", "Rate_gini",
        ],
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for signal in ("LFP", "SNE", "MUA"):
            signal_frame = frame.loc[frame["Signal"].eq(signal)].copy()
            if signal_frame.empty:
                continue
            signal_frame.drop(columns=["Signal"], inplace=True)
            signal_frame.dropna(axis=1, how="all", inplace=True)
            leading = [column for column in preferred[signal] if column in signal_frame.columns]
            trailing = [column for column in signal_frame.columns if column not in leading]
            signal_frame = signal_frame[leading + trailing]
            signal_frame.to_excel(writer, sheet_name=signal, index=False)

            worksheet = writer.sheets[signal]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_view.showGridLines = False
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
                cell.alignment = cell.alignment.copy(wrap_text=True, vertical="center")
            worksheet.row_dimensions[1].height = 30
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(max(map(len, values)) + 2, 10), 34)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
                for cell in column_cells[1:]:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"


def analyze_recording(
    path: Path,
    output_root: Path,
    config: AnalysisConfig,
    log: Log,
    electrode_results: dict[str, list[dict]] | None = None,
    propagation_summary: dict[str, list[pd.DataFrame]] | None = None,
) -> list[dict]:
    names, groups = load_groups(path)
    sr, duration, streams, h5 = _read_streams(path, config.well)
    output = output_root / _safe_name(path.stem)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    try:
        group_map = dict(zip(names, groups))
        all_electrodes = np.unique(np.concatenate(groups))
        selected_groups = {"Full": all_electrodes, **group_map}
        for signal in ("LFP", "MUA"):
            if not config.network_activity_enabled:
                break
            if signal == "LFP" and not config.analyze_lfp:
                continue
            if signal == "MUA" and not config.analyze_mua:
                continue
            if signal not in streams:
                log(f"  {signal}: stream not present; skipped")
                continue
            channels, times = streams[signal]
            rate_scale = 60.0 if signal == "LFP" else 1.0
            threshold = config.active_threshold_lfp if signal == "LFP" else config.active_threshold_mua
            signal_metrics = config.lfp_metrics if signal == "LFP" else config.mua_metrics
            amplitudes = _event_amplitudes(h5, config.well, signal, channels) if "amplitude" in signal_metrics else {}
            if config.generate_raster:
                _save_raster(
                    times, channels, output / f"{path.stem}_{signal}_raster",
                    config.raster_start_s, config.raster_end_s, config.raster_formats,
                )
            for group_name, electrodes in selected_groups.items():
                mask = np.isin(channels, electrodes)
                gch, gt = channels[mask], times[mask]
                unique, counts = np.unique(gch, return_counts=True)
                rates = counts / duration * rate_scale if duration else np.zeros_like(counts, dtype=float)
                active_mask = rates >= threshold
                active_rates = rates[active_mask]
                if "electrode_frequencies" in signal_metrics and electrode_results is not None:
                    frequency_column = "Frequency_FP_per_min" if signal == "LFP" else "Frequency_Hz"
                    for electrode, frequency in zip(unique, rates):
                        electrode_results[signal].append({
                            "File": path.name,
                            "Group": group_name,
                            "Electrode_ID": int(electrode),
                            frequency_column: float(frequency),
                        })
                row = {"File": path.name, "Signal": signal, "Group": group_name, "Duration_s": duration, "Sampling_rate_Hz": sr}
                unit = "FP/min" if signal == "LFP" else "Hz"
                if "frequency_distribution_stats" in signal_metrics:
                    row["Rate_skewness"] = float(skew(active_rates)) if len(active_rates) > 2 else np.nan
                    row["Rate_kurtosis"] = float(kurtosis(active_rates)) if len(active_rates) > 3 else np.nan
                    sorted_rates = np.sort(active_rates)
                    n = len(sorted_rates)
                    row["Rate_gini"] = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_rates) / (n * np.sum(sorted_rates))) if n and np.sum(sorted_rates) else np.nan
                if "frequency" in signal_metrics:
                    for label, value in _aggregate(active_rates, config.aggregations).items():
                        row[f"{label}_Frequency_{unit}"] = value
                if "cv_iei" in signal_metrics:
                    for label, value in _aggregate(_per_electrode_cv(gt, gch), config.aggregations).items():
                        row[f"{label}_CV_IEI"] = value
                if "active_electrodes" in signal_metrics:
                    row["Active_electrodes"] = int(active_mask.sum())
                if "amplitude" in signal_metrics:
                    vals = [amplitudes.get(int(ch), np.nan) for ch in unique[active_mask]]
                    for label, value in _aggregate(vals, config.aggregations).items():
                        row[f"{label}_Amplitude_pA"] = value
                base = _safe_name(f"{path.stem}_{signal}_{group_name}")
                if "frequency_distribution_comparison" in config.plots:
                    _save_frequency_distribution_comparison(
                        active_rates,
                        output / f"{base}_Freq_Distribution_Comparison.png",
                        f"{signal} — {group_name}",
                    )
                rows.append(row)

        if config.network_activity_enabled and config.analyze_lfp and config.analyze_mua and "mua_lfp_ratio" in config.mua_metrics:
            lfp_rows = {r["Group"]: r for r in rows if r["Signal"] == "LFP"}
            for row in (r for r in rows if r["Signal"] == "MUA"):
                lfp_row = lfp_rows.get(row["Group"], {})
                for agg in config.aggregations:
                    label = {"mean": "Mean", "geometric_mean": "Geometric_mean", "median": "Median"}[agg]
                    lfp_rate = lfp_row.get(f"{label}_Frequency_FP/min")
                    mua_rate = row.get(f"{label}_Frequency_Hz")
                    row[f"{label}_MUA_LFP_ratio"] = (mua_rate * 60.0 / lfp_rate) if mua_rate is not None and lfp_rate else np.nan

        if config.network_activity_enabled and config.analyze_lfp and "LFP" in streams:
            channels, times = streams["LFP"]
            requested = ["Full", *group_map] if "All groups" in config.sne_groups else config.sne_groups
            for sne_group in requested:
                if sne_group == "Full":
                    sne_electrodes = all_electrodes
                elif sne_group in group_map:
                    sne_electrodes = group_map[sne_group]
                else:
                    log(f"  SNE group '{sne_group}' is not present; skipped")
                    continue
                direct_events, hist_data = _detect_snes_histogram(times, channels, sne_electrodes, config)
                kdl_events, durations, kde_data = _detect_snes(times, channels, sne_electrodes, config)
                regularity = _regularity(kdl_events, config.fano_window_s)
                sne_row = {"File": path.name, "Signal": "SNE", "Group": sne_group}
                sne_frequency = float(len(direct_events) / duration * 60) if duration else np.nan
                sne_kdl_frequency = float(len(kdl_events) / duration * 60) if duration else np.nan
                if "sne_frequency" in config.sne_metrics:
                    sne_row["SNE_frequency_per_min"] = sne_frequency
                    sne_row["SNE_count"] = len(direct_events)
                if "sne_kdl_frequency" in config.sne_metrics:
                    sne_row["SNE_KDL_frequency_per_min"] = sne_kdl_frequency
                    sne_row["SNE_KDL_count"] = len(kdl_events)
                if "sne_duration" in config.sne_metrics:
                    sne_row["SNE_mean_duration_s"] = float(np.mean(durations)) if durations else np.nan
                if "cv_iei" in config.sne_metrics:
                    sne_row["Mean_CV_IEI"] = regularity["cv_iei"]
                if "fano" in config.sne_metrics:
                    sne_row["FANO"] = regularity["fano"]
                if "lfp_sne_ratio" in config.lfp_metrics:
                    lfp_row = next((r for r in rows if r["Signal"] == "LFP" and r["Group"] == sne_group), None)
                    if lfp_row:
                        for agg in config.aggregations:
                            label = {"mean": "Mean", "geometric_mean": "Geometric_mean", "median": "Median"}[agg]
                            lfp_rate = lfp_row.get(f"{label}_Frequency_FP/min")
                            sne_row[f"{label}_LFP_SNE_ratio"] = lfp_rate / sne_frequency if lfp_rate is not None and sne_frequency else np.nan
                if "lfp_sne_kdl_ratio" in config.lfp_metrics:
                    lfp_row = next((r for r in rows if r["Signal"] == "LFP" and r["Group"] == sne_group), None)
                    if lfp_row:
                        for agg in config.aggregations:
                            label = {"mean": "Mean", "geometric_mean": "Geometric_mean", "median": "Median"}[agg]
                            lfp_rate = lfp_row.get(f"{label}_Frequency_FP/min")
                            sne_row[f"{label}_LFP_SNE_KDL_ratio"] = lfp_rate / sne_kdl_frequency if lfp_rate is not None and sne_kdl_frequency else np.nan
                base = _safe_name(f"{path.stem}_{sne_group}")
                if "psd_peaks" in config.sne_metrics:
                    peaks = _save_psd(regularity["iei"], output / f"{base}_SNE_KDL_PSD.png")
                    if "psd_peaks" in config.sne_metrics:
                        sne_row["PSD_peaks"] = json.dumps(peaks)
                if "acf_peaks" in config.sne_metrics:
                    peaks = _save_acf(regularity["iei"], output / f"{base}_SNE_KDL_ACF.png", config.acf_max_lag)
                    if "acf_peaks" in config.sne_metrics:
                        sne_row["ACF_peaks"] = json.dumps(peaks)
                if "sync" in config.plots and hist_data is not None:
                    hist, edges, peaks, minimum = hist_data
                    centers = (edges[:-1] + edges[1:]) / 2
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.bar(centers, hist, width=np.diff(edges), alpha=.6, color="green")
                    ax.axhline(minimum, color="red", label=f"Threshold: {minimum}")
                    ax.scatter(centers[peaks], hist[peaks], color="black", s=20)
                    ax.set(title="Synchronous Network Events", xlabel="Time (s)", ylabel="Event count")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(output / f"{base}_Sync.png", dpi=180)
                    plt.close(fig)
                if "sync_kde" in config.plots and kde_data is not None:
                    grid, density, peaks, threshold = kde_data
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(grid, density)
                    ax.axhline(threshold, color="red", linestyle="--")
                    ax.scatter(grid[peaks], density[peaks], color="black", s=20)
                    ax.set(title="SNE KDE detection", xlabel="Time (s)", ylabel="Density")
                    fig.tight_layout()
                    fig.savefig(output / f"{base}_Sync_KDE.png", dpi=180)
                    plt.close(fig)
                rows.append(sne_row)
        if config.propagation_enabled and config.analyze_lfp and "LFP" in streams:
            channels, times = streams["LFP"]
            propagation = run_propagation_analysis(
                path.stem, output, times, channels, duration, group_map, config, log
            )
            if propagation_summary is not None:
                for sheet_name, frame in propagation.items():
                    propagation_summary.setdefault(sheet_name, []).append(frame)
    finally:
        h5.close()
    if rows:
        pd.DataFrame(rows).to_excel(output / f"{path.stem}_results.xlsx", index=False)
    return rows


def run_analysis(config: AnalysisConfig, log: Log = print) -> Path:
    recordings = discover_recordings(config.inputs, config.recursive)
    if not recordings:
        raise FileNotFoundError("No .brx or .bxr recordings were found in the selected input.")
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "analysis_config.json")
    all_rows: list[dict] = []
    electrode_results: dict[str, list[dict]] = {"LFP": [], "MUA": []}
    propagation_summary: dict[str, list[pd.DataFrame]] = {}
    errors: list[str] = []
    completed = 0
    log(f"Found {len(recordings)} recording(s).")
    for index, recording in enumerate(recordings, 1):
        log(f"[{index}/{len(recordings)}] {recording}")
        try:
            all_rows.extend(analyze_recording(
                recording, output, config, log, electrode_results, propagation_summary
            ))
            completed += 1
            log("  completed")
        except Exception as exc:
            errors.append(f"{recording}\n{exc}\n{traceback.format_exc()}")
            log(f"  ERROR: {exc}")
    if all_rows:
        _write_signal_summary(all_rows, output / "Summary_Results.xlsx")
    if propagation_summary:
        write_propagation_summary(
            propagation_summary, output / "Signal Propagation Analysis summary.xlsx"
        )
    for signal, filename in (
        ("LFP", "LFP_Summary_Electrode_Frequencies.xlsx"),
        ("MUA", "MUA_Summary_Electrode_Frequencies.xlsx"),
    ):
        records = electrode_results[signal]
        if not records:
            continue
        frame = pd.DataFrame(records)
        with pd.ExcelWriter(output / filename) as writer:
            used_names: set[str] = set()
            for group_name, group_frame in frame.groupby("Group", sort=False):
                base = "".join(ch if ch not in r"[]:*?/\\" else "_" for ch in str(group_name))[:31] or "Group"
                sheet_name = base
                suffix = 2
                while sheet_name.lower() in used_names:
                    tail = f"_{suffix}"
                    sheet_name = base[:31 - len(tail)] + tail
                    suffix += 1
                used_names.add(sheet_name.lower())
                group_frame.to_excel(writer, sheet_name=sheet_name, index=False)
    if errors:
        (output / "error_log.txt").write_text("\n\n".join(errors), encoding="utf-8")
    if completed == 0:
        raise RuntimeError(f"All {len(recordings)} recording(s) failed. See error_log.txt.")
    log(f"Finished. Results: {output}")
    return output
