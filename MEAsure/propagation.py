from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _channel_coordinates(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert zero-based 4096-channel IDs to the source script's 64×64 grid."""
    channels = np.asarray(channels, dtype=int)
    return channels // 64, channels % 64


def detect_event_windows(
    times: np.ndarray,
    channels: np.ndarray,
    electrodes: np.ndarray,
    minimum_active_percent: float,
    boundary_percent: float,
    bins: int,
) -> list[dict]:
    mask = np.isin(channels, electrodes)
    selected_times = np.asarray(times[mask], dtype=float)
    selected_channels = np.asarray(channels[mask], dtype=int)
    if len(selected_times) < 2:
        return []
    hist, edges = np.histogram(selected_times, bins=max(2, int(bins)))
    minimum = max(1, int(np.ceil(len(np.unique(electrodes)) * minimum_active_percent / 100.0)))
    padded_peaks, _ = find_peaks(np.r_[0, hist, 0], height=minimum)
    peaks = padded_peaks - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(hist))]
    boundary = hist.max() * boundary_percent / 100.0
    events = []
    for peak in peaks:
        left, right = int(peak), int(peak + 1)
        if boundary_percent > 0:
            while left > 0 and hist[left] >= boundary:
                left -= 1
            while right < len(hist) and hist[right] >= boundary:
                right += 1
        start, end = float(edges[left]), float(edges[min(right, len(edges) - 1)])
        event_mask = (selected_times >= start) & (selected_times < end)
        event_channels = selected_channels[event_mask]
        if np.unique(event_channels).size < minimum:
            continue
        order = np.argsort(selected_times[event_mask])
        events.append({
            "start_s": start,
            "end_s": end,
            "peak_s": float((edges[peak] + edges[peak + 1]) / 2),
            "times": selected_times[event_mask][order],
            "channels": event_channels[order],
        })
    return events


def chronological_centroids(event: dict, step_s: float, tolerance_s: float) -> np.ndarray:
    """Return ordered rows of (time_s, row, column); never converts through an occupancy matrix."""
    times = event["times"]
    channels = event["channels"]
    if len(times) == 0:
        return np.empty((0, 3), dtype=float)
    step_s = max(float(step_s), np.finfo(float).eps)
    tolerance_s = max(float(tolerance_s), 0.0)
    samples = np.arange(float(times[0]), float(times[-1]) + step_s / 2, step_s)
    rows, cols = _channel_coordinates(channels)
    trajectory = []
    for sample in samples:
        mask = np.abs(times - sample) <= tolerance_s
        if np.any(mask):
            trajectory.append((sample, float(np.mean(rows[mask])), float(np.mean(cols[mask]))))
    return np.asarray(trajectory, dtype=float).reshape(-1, 3)


def propagation_metrics(trajectory: np.ndarray, event_index: int, pitch_cm: float) -> dict:
    result = {
        "Event": event_index,
        "Duration_s": np.nan,
        "Total_distance_cm": 0.0,
        "Maximum_distance_from_origin_cm": 0.0,
        "Total_speed_cm_per_s": 0.0,
        "Expansion_distance_cm": 0.0,
        "Expansion_speed_cm_per_s": 0.0,
        "Retraction_distance_cm": 0.0,
        "Retraction_speed_cm_per_s": 0.0,
        "Centroid_samples": int(len(trajectory)),
    }
    if len(trajectory) == 0:
        return result
    duration = float(trajectory[-1, 0] - trajectory[0, 0])
    result["Duration_s"] = duration
    if len(trajectory) < 2:
        return result
    xy = trajectory[:, 1:3]
    segment_distances = np.linalg.norm(np.diff(xy, axis=0), axis=1) * pitch_cm
    radial = np.linalg.norm(xy - xy[0], axis=1) * pitch_cm
    maximum_index = int(np.argmax(radial))
    total_distance = float(segment_distances.sum())
    expansion_distance = float(segment_distances[:maximum_index].sum())
    retraction_distance = float(segment_distances[maximum_index:].sum())
    expansion_time = float(trajectory[maximum_index, 0] - trajectory[0, 0])
    retraction_time = float(trajectory[-1, 0] - trajectory[maximum_index, 0])
    result.update({
        "Total_distance_cm": total_distance,
        "Maximum_distance_from_origin_cm": float(radial[maximum_index]),
        "Total_speed_cm_per_s": total_distance / duration if duration > 0 else 0.0,
        "Expansion_distance_cm": expansion_distance,
        "Expansion_speed_cm_per_s": expansion_distance / expansion_time if expansion_time > 0 else 0.0,
        "Retraction_distance_cm": retraction_distance,
        "Retraction_speed_cm_per_s": retraction_distance / retraction_time if retraction_time > 0 else 0.0,
    })
    return result


def _trajectory_groups(trajectory: np.ndarray, route_groups: dict[str, np.ndarray]) -> list[str]:
    memberships = {name: set(np.asarray(group, dtype=int).tolist()) for name, group in route_groups.items()}
    sequence = []
    for _, row, col in trajectory:
        channel = int(np.clip(round(row), 0, 63) * 64 + np.clip(round(col), 0, 63))
        name = next((group_name for group_name, members in memberships.items() if channel in members), "OUT")
        sequence.append(name)
    return sequence


def _deduplicate_consecutive(sequence: list[str]) -> list[str]:
    return [value for index, value in enumerate(sequence) if index == 0 or value != sequence[index - 1]]


def route_tables(trajectories: list[np.ndarray], route_groups: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    raw_sequences = [_trajectory_groups(trajectory, route_groups) for trajectory in trajectories]
    sequences = [[name for name in sequence if name != "OUT"] for sequence in raw_sequences]
    valid_sequences = [sequence for sequence in sequences if sequence]
    names = list(route_groups)

    per_event = []
    for event_index, sequence in enumerate(valid_sequences):
        counts = Counter(sequence)
        total = len(sequence)
        for name in names:
            per_event.append({"Event": event_index, "Group": name, "Percentage": counts[name] / total * 100})
    per_event_frame = pd.DataFrame(per_event)
    if per_event_frame.empty:
        average = pd.DataFrame(columns=["Group", "Average_occurrence_percentage"])
    else:
        average = (
            per_event_frame.groupby("Group", sort=False)["Percentage"]
            .mean().reindex(names, fill_value=0).rename("Average_occurrence_percentage").reset_index()
        )

    overall_counts = Counter(name for sequence in valid_sequences for name in sequence)
    overall_total = sum(overall_counts.values())
    overall = pd.DataFrame({
        "Group": names,
        "Overall_occurrence_percentage": [
            overall_counts[name] / overall_total * 100 if overall_total else 0.0 for name in names
        ],
    })

    routes = [_deduplicate_consecutive(sequence) for sequence in valid_sequences]
    routes = [route for route in routes if route]
    denominator = len(routes)
    full_counts = Counter(tuple(route) for route in routes)
    full = pd.DataFrame([
        {"Route": " → ".join(route), "Occurrence_percentage": count / denominator * 100}
        for route, count in full_counts.most_common()
    ])

    sub_counts = Counter()
    for route in routes:
        unique_subroutes = {
            tuple(route[start:end])
            for start in range(len(route))
            for end in range(start + 2, len(route) + 1)
        }
        sub_counts.update(unique_subroutes)
    sub = pd.DataFrame([
        {"Sub_route": " → ".join(route), "Occurrence_percentage": count / denominator * 100}
        for route, count in sub_counts.most_common() if count > 1
    ])

    origins = Counter(route[0] for route in routes)
    endings = Counter(route[-1] for route in routes)
    origin = pd.DataFrame({
        "Group": names,
        "Signal_origin_percentage": [origins[name] / denominator * 100 if denominator else 0.0 for name in names],
    })
    ending = pd.DataFrame({
        "Group": names,
        "Signal_ending_percentage": [endings[name] / denominator * 100 if denominator else 0.0 for name in names],
    })
    return {
        "Average_Occurrence": average,
        "Overall_Occurrence": overall,
        "Full_Routes": full,
        "Sub_Routes": sub,
        "Signal_Origin": origin,
        "Signal_End": ending,
    }


def _plot_lfp_heatmap(times: np.ndarray, channels: np.ndarray, duration: float, path: Path) -> None:
    heatmap = np.zeros((64, 64), dtype=float)
    unique, counts = np.unique(channels, return_counts=True)
    rows, cols = _channel_coordinates(unique)
    valid = (rows >= 0) & (rows < 64) & (cols >= 0) & (cols < 64)
    heatmap[rows[valid], cols[valid]] = counts[valid] / duration if duration > 0 else 0.0
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(heatmap, cmap="viridis", origin="lower")
    ax.set(title="LFP 64 × 64 Electrode Frequency Map", xlabel="Column", ylabel="Row")
    fig.colorbar(image, ax=ax, label="Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cats(trajectories: list[np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for event_index, trajectory in enumerate(trajectories):
        if len(trajectory):
            ax.plot(trajectory[:, 2], trajectory[:, 1], marker="o", markersize=2, linewidth=1, alpha=.65, label=f"Event {event_index}")
    ax.set(xlim=(0, 63), ylim=(63, 0), title="Centre-of-Activity Trajectories", xlabel="Column", ylabel="Row")
    if 0 < len(trajectories) <= 12:
        ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_weighted_cats(trajectories: list[np.ndarray], path: Path) -> None:
    counts = np.zeros((64, 64), dtype=float)
    for trajectory in trajectories:
        for _, row, col in trajectory:
            counts[int(np.clip(round(row), 0, 63)), int(np.clip(round(col), 0, 63))] += 1
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(counts, cmap="inferno", origin="lower")
    ax.set(title="Combined Weighted CATS", xlabel="Column", ylabel="Row")
    fig.colorbar(image, ax=ax, label="Centroid occurrence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _add_nonzero_average(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    average = {"Event": "average"}
    for column in frame.columns:
        if column == "Event":
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        nonzero = numeric[(numeric != 0) & numeric.notna()]
        average[column] = float(nonzero.mean()) if len(nonzero) else np.nan
    return pd.concat([frame, pd.DataFrame([average])], ignore_index=True)


def _format_workbook(writer: pd.ExcelWriter) -> None:
    for worksheet in writer.sheets.values():
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.sheet_view.showGridLines = False
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True, color="FFFFFF")
            cell.fill = cell.fill.copy(fill_type="solid", fgColor="7030A0")
            cell.alignment = cell.alignment.copy(wrap_text=True, vertical="center")
        worksheet.row_dimensions[1].height = 30
        for cells in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else "" for cell in cells]
            worksheet.column_dimensions[cells[0].column_letter].width = min(max(max(map(len, values)) + 2, 11), 36)
            for cell in cells[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"


def run_propagation_analysis(
    recording_name: str,
    output: Path,
    times: np.ndarray,
    channels: np.ndarray,
    duration: float,
    group_map: dict[str, np.ndarray],
    config,
    log,
) -> dict[str, pd.DataFrame]:
    region_results: dict[str, pd.DataFrame] = {}
    all_trajectories: list[np.ndarray] = []
    hp_trajectories: list[np.ndarray] = []
    for region in ("HP", "CTX"):
        if region not in config.propagation_regions:
            continue
        if region not in group_map:
            log(f"  Propagation: group '{region}' not present; skipped")
            continue
        events = detect_event_windows(
            times, channels, group_map[region],
            config.propagation_min_active_percent,
            config.propagation_boundary_percent,
            config.propagation_bins,
        )
        trajectories = [
            chronological_centroids(event, config.centroid_step_s, config.centroid_tolerance_s)
            for event in events
        ]
        trajectories = [trajectory for trajectory in trajectories if len(trajectory)]
        metrics = [
            propagation_metrics(trajectory, index, config.electrode_pitch_um / 10000.0)
            for index, trajectory in enumerate(trajectories)
        ]
        frame = pd.DataFrame(metrics)
        metric_columns = {
            "duration": "Duration_s",
            "total_distance": "Total_distance_cm",
            "maximum_distance": "Maximum_distance_from_origin_cm",
            "total_speed": "Total_speed_cm_per_s",
            "expansion_distance": "Expansion_distance_cm",
            "expansion_speed": "Expansion_speed_cm_per_s",
            "retraction_distance": "Retraction_distance_cm",
            "retraction_speed": "Retraction_speed_cm_per_s",
        }
        selected_columns = ["Event"] + [
            column for key, column in metric_columns.items() if key in config.propagation_metrics and column in frame.columns
        ]
        if "Centroid_samples" in frame.columns:
            selected_columns.append("Centroid_samples")
        frame = frame[selected_columns] if not frame.empty else pd.DataFrame(columns=selected_columns)
        region_results[f"{region}_Propagation"] = _add_nonzero_average(frame)
        all_trajectories.extend(trajectories)
        if region == "HP":
            hp_trajectories = trajectories

    route_group_map = {name: group_map[name] for name in config.route_group_names if name in group_map}
    route_results = route_tables(hp_trajectories, route_group_map) if route_group_map else {}
    route_sheet_keys = {
        "average_occurrence": "Average_Occurrence",
        "overall_occurrence": "Overall_Occurrence",
        "full_routes": "Full_Routes",
        "sub_routes": "Sub_Routes",
        "signal_origin": "Signal_Origin",
        "signal_end": "Signal_End",
    }
    route_results = {
        sheet: route_results[sheet]
        for key, sheet in route_sheet_keys.items()
        if key in config.route_metrics and sheet in route_results
    }
    sheets = {**region_results, **route_results}
    if not sheets and not config.propagation_plots:
        return {}
    workbook = output / f"{recording_name}_Signal_Propagation.xlsx"
    if sheets:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            _format_workbook(writer)

    if "lfp_heatmap" in config.propagation_plots:
        _plot_lfp_heatmap(times, channels, duration, output / f"{recording_name}_LFP_64x64_Heatmap.png")
    if "cats" in config.propagation_plots:
        _plot_cats(all_trajectories, output / f"{recording_name}_CATS.png")
    if "weighted_cats" in config.propagation_plots:
        _plot_weighted_cats(all_trajectories, output / f"{recording_name}_Combined_Weighted_CATS.png")
    summary: dict[str, pd.DataFrame] = {}
    for sheet_name, frame in region_results.items():
        average = frame.loc[frame["Event"].eq("average")].copy() if "Event" in frame.columns else pd.DataFrame()
        if average.empty:
            continue
        average.drop(columns=["Event"], inplace=True)
        average.insert(0, "File", recording_name)
        summary[sheet_name] = average
    for sheet_name, frame in route_results.items():
        combined = frame.copy()
        combined.insert(0, "File", recording_name)
        summary[sheet_name] = combined
    return summary


def write_propagation_summary(collected: dict[str, list[pd.DataFrame]], path: Path) -> None:
    sheets = {
        sheet_name: pd.concat(frames, ignore_index=True, sort=False)
        for sheet_name, frames in collected.items() if frames
    }
    if not sheets:
        return
    order = [
        "HP_Propagation", "CTX_Propagation", "Average_Occurrence", "Overall_Occurrence",
        "Full_Routes", "Sub_Routes", "Signal_Origin", "Signal_End",
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in order:
            if sheet_name in sheets:
                frame = sheets[sheet_name]
                if sheet_name != "HP_Propagation" and "File" in frame.columns and not frame.empty:
                    rows = []
                    previous = None
                    for _, row in frame.iterrows():
                        filename = row["File"]
                        if previous is not None and filename != previous:
                            rows.append({column: None for column in frame.columns})
                        rows.append(row.to_dict())
                        previous = filename
                    frame = pd.DataFrame(rows, columns=frame.columns)
                frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        _format_workbook(writer)
