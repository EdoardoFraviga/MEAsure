from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

try:
    from .fieldmap import load_fp_rate_map
except ImportError:
    from fieldmap import load_fp_rate_map


MANUAL_GROUP = "Manual Selection"


def read_json_groups(path: str | Path) -> dict[str, set[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = {
        str(group["UserDefinedName"]): {int(value) for value in group.get("PixelIndexes", [])}
        for group in payload.get("Groups", [])
    }
    if not groups:
        raise ValueError("The matching JSON contains no electrode groups.")
    return groups


def matching_companion(source: str | Path, extensions: tuple[str, ...]) -> Path | None:
    source = Path(source)
    wanted = {extension.lower() for extension in extensions}
    for candidate in source.parent.iterdir():
        if candidate.is_file() and candidate.stem.lower() == source.stem.lower() and candidate.suffix.lower() in wanted:
            return candidate
    return None


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value).strip(" .")
    return cleaned or "Group"


def fp_event_counts(path: str | Path, well: str) -> dict[int, int]:
    """Count waveform-bearing FP events using the original script's BXR paths."""
    with h5py.File(path, "r") as handle:
        if well not in handle:
            raise KeyError(f"Well '{well}' was not found in {Path(path).name}.")
        group = handle[well]
        for dataset in ("FpChIdxs", "FpForms"):
            if dataset not in group:
                raise KeyError(f"'{well}/{dataset}' was not found in {Path(path).name}.")
        channels = np.asarray(group["FpChIdxs"]).reshape(-1).astype(int)
        values, counts = np.unique(channels, return_counts=True)
    return {int(channel): int(count) for channel, count in zip(values, counts)}


def extract_average_waveform(path: str | Path, well: str, electrodes: set[int]):
    with h5py.File(path, "r") as handle:
        if well not in handle:
            raise KeyError(f"Well '{well}' was not found.")
        group = handle[well]
        channels = np.asarray(group["FpChIdxs"]).reshape(-1).astype(int)
        forms = np.asarray(group["FpForms"]).reshape(-1)
        wavelength = int(np.asarray(group["FpForms"].attrs["Wavelength"]).reshape(-1)[0])
        sampling_rate = float(np.asarray(handle.attrs["SamplingRate"]).reshape(-1)[0])
        min_d = float(np.asarray(handle.attrs["MinDigitalValue"]).reshape(-1)[0])
        max_d = float(np.asarray(handle.attrs["MaxDigitalValue"]).reshape(-1)[0])
        min_a = float(np.asarray(handle.attrs["MinAnalogValue"]).reshape(-1)[0])
        max_a = float(np.asarray(handle.attrs["MaxAnalogValue"]).reshape(-1)[0])
    factor = (max_a - min_a) / (max_d - min_d)
    offset = min_a - factor * min_d
    waveforms = []
    for event_index in np.flatnonzero(np.isin(channels, list(electrodes))):
        raw = forms[event_index * wavelength:(event_index + 1) * wavelength].astype(np.int16)
        nonzero = np.flatnonzero(raw)
        if len(nonzero):
            waveforms.append(offset + factor * raw[:nonzero[-1] + 1])
    if not waveforms:
        raise ValueError("No FP waveforms were found for the selected electrodes.")
    width = max(map(len, waveforms))
    aligned = np.full((len(waveforms), width), np.nan)
    for row, waveform in enumerate(waveforms):
        aligned[row, :len(waveform)] = waveform
    return (
        np.arange(width) / sampling_rate,
        np.nanmean(aligned, axis=0),
        np.nanstd(aligned, axis=0),
        len(waveforms),
    )


def save_waveform_plot(
    source: Path, output_dir: Path, well: str, group_name: str,
    electrodes: set[int], formats: set[str],
) -> list[Path]:
    time, average, deviation, event_count = extract_average_waveform(source, well, electrodes)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(time, average, color="#2F5597", label="Average FP waveform")
    axis.fill_between(time, average - deviation, average + deviation, color="#5B9BD5", alpha=.3, label="±1 SD")
    axis.set(
        title=f"Average FP Waveform — {group_name}\n{len(electrodes)} electrodes; N = {event_count} events",
        xlabel="Time (s)", ylabel="Voltage (µV)",
    )
    axis.legend()
    axis.grid(alpha=.2)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    base = output_dir / f"{source.stem}_{safe_filename(group_name)}_Average_FP_Waveform"
    for fmt in formats:
        suffix = ".emf.svg" if fmt == "emf.svg" else f".{fmt}"
        target = Path(str(base) + suffix)
        figure.savefig(target, format="svg" if fmt == "emf.svg" else fmt, dpi=300, bbox_inches="tight")
        produced.append(target)
    plt.close(figure)
    return produced


class WaveformFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=16)
        self.files: list[str] = []
        self.current_file: Path | None = None
        self.current_bx_path: Path | None = None
        self.original_groups: dict[str, set[int]] = {}
        self.refined_groups: dict[str, set[int]] = {}
        self.manual_electrodes: set[int] = set()
        self.rate_map: np.ndarray | None = None
        self.fp_counts: dict[int, int] = {}
        self.map_patches: list[Rectangle] = []
        self._build()

    def _build(self):
        ttk.Label(self, text="Waveform extraction", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text="A BRW identifies the recording; FP waveforms and the activity map are read from its matching BRX/BXR.",
        ).pack(anchor="w", pady=(2, 12))

        top = ttk.Frame(self)
        top.pack(fill="x")
        files_box = ttk.LabelFrame(top, text="1 · BRW files", style="Section.TLabelframe", padding=10)
        files_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.file_list = tk.Listbox(files_box, height=6, selectmode="extended", exportselection=False)
        self.file_list.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.file_list.bind("<<ListboxSelect>>", self._file_selected)
        ttk.Button(files_box, text="Add BRW file(s)…", command=self.add_files).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(files_box, text="Remove", command=self.remove_files).grid(row=1, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(files_box, text="Clear", command=self.clear_files).grid(row=2, column=1, sticky="new", padx=(6, 0))
        files_box.columnconfigure(0, weight=1)

        export_box = ttk.LabelFrame(top, text="2 · Settings and export", style="Section.TLabelframe", padding=10)
        export_box.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.well_var = tk.StringVar(value="Well_A1")
        ttk.Label(export_box, text="Well").grid(row=0, column=0, sticky="w")
        ttk.Entry(export_box, textvariable=self.well_var, width=16).grid(row=0, column=1, sticky="w")
        self.output_var = tk.StringVar(value=str(Path.home() / "Desktop" / "MEA_Waveforms"))
        ttk.Label(export_box, text="Output folder").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(export_box, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(export_box, text="Browse…", command=self.choose_output).grid(row=1, column=2, padx=(6, 0), pady=(8, 0))
        ttk.Label(export_box, text="Output formats").grid(row=2, column=0, sticky="w", pady=(10, 0))
        format_frame = ttk.Frame(export_box)
        format_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=(10, 0))
        self.format_vars = {}
        for column, (key, label) in enumerate((("png", ".png"), ("svg", ".svg"), ("emf.svg", ".emf.svg"))):
            var = tk.BooleanVar(value=True)
            self.format_vars[key] = var
            ttk.Checkbutton(format_frame, text=label, variable=var).grid(row=0, column=column, padx=(0, 8))
        export_box.columnconfigure(1, weight=1)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=10)
        map_box = ttk.LabelFrame(body, text="Matching BRX/BXR LFP activity map", padding=6)
        map_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.figure = Figure(figsize=(7, 5))
        self.axis = self.figure.add_axes((0.09, 0.10, 0.74, 0.82))
        self.color_axis = self.figure.add_axes((0.87, 0.10, 0.04, 0.82))
        self.color_axis.set_axis_off()
        self.axis.set(title="Select a BRW file", xlabel="Column", ylabel="Row")
        self.canvas = FigureCanvasTkAgg(self.figure, master=map_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._map_click)

        groups_box = ttk.LabelFrame(body, text="Matching JSON groups", padding=10)
        groups_box.pack(side="left", fill="both", padx=(5, 0))
        ttk.Label(
            groups_box,
            text="Select one or more groups.\nActivate Manual Selection to choose individual\nJSON electrodes, or refine a named group.",
        ).pack(anchor="w")
        self.group_list = tk.Listbox(groups_box, width=35, height=14, selectmode="multiple", exportselection=False)
        self.group_list.pack(fill="both", expand=True, pady=6)
        self.group_list.bind("<<ListboxSelect>>", lambda _event: self._redraw_map())
        ttk.Button(groups_box, text="Restore active group", command=self.restore_active_group).pack(fill="x")
        self.companion_var = tk.StringVar(value="No BRW selected.")
        ttk.Label(groups_box, textvariable=self.companion_var, wraplength=270).pack(anchor="w", pady=(10, 0))

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        self.run_button = ttk.Button(controls, text="Extract selected group waveforms", command=self.start)
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=12)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", pady=(8, 0))

    def add_files(self):
        added = []
        for path in filedialog.askopenfilenames(filetypes=[("3Brain BRW", "*.brw"), ("All files", "*.*")]):
            if path not in self.files:
                self.files.append(path)
                self.file_list.insert("end", path)
                added.append(path)
        if added:
            index = self.files.index(added[0])
            self.file_list.selection_clear(0, "end")
            self.file_list.selection_set(index)
            self.file_list.activate(index)
            self._load_companions(Path(added[0]))

    def remove_files(self):
        for index in reversed(self.file_list.curselection()):
            del self.files[index]
            self.file_list.delete(index)
        self._clear_preview()

    def clear_files(self):
        self.files.clear()
        self.file_list.delete(0, "end")
        self._clear_preview()

    def choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def _file_selected(self, _event=None):
        selection = self.file_list.curselection()
        if selection:
            self._load_companions(Path(self.files[selection[-1]]))

    def _clear_preview(self):
        self.current_file, self.current_bx_path, self.rate_map = None, None, None
        self.original_groups.clear()
        self.refined_groups.clear()
        self.manual_electrodes.clear()
        self.fp_counts.clear()
        self.group_list.delete(0, "end")
        self.axis.clear()
        self.color_axis.clear()
        self.color_axis.set_axis_off()
        self.axis.set(title="Select a BRW file", xlabel="Column", ylabel="Row")
        self.canvas.draw_idle()
        self.companion_var.set("No BRW selected.")

    def _load_companions(self, source: Path):
        json_path = matching_companion(source, (".json",))
        bx_path = matching_companion(source, (".brx", ".bxr"))
        if json_path is None or bx_path is None:
            missing = []
            if json_path is None:
                missing.append(f"{source.stem}.json")
            if bx_path is None:
                missing.append(f"{source.stem}.brx/.bxr")
            self._clear_preview()
            self.current_file = source
            self.companion_var.set("Missing beside BRW: " + ", ".join(missing))
            return
        try:
            groups = read_json_groups(json_path)
            rate_map, _duration = load_fp_rate_map(bx_path, self.well_var.get().strip())
            counts = fp_event_counts(bx_path, self.well_var.get().strip())
        except Exception as exc:
            self._clear_preview()
            self.current_file = source
            self.companion_var.set(f"Cannot load companions: {exc}")
            return
        self.current_file = source
        self.current_bx_path = bx_path
        self.original_groups = groups
        self.refined_groups = {name: set(electrodes) for name, electrodes in groups.items()}
        self.manual_electrodes = set()
        self.rate_map = rate_map
        self.fp_counts = counts
        self.group_list.delete(0, "end")
        self.group_list.insert("end", f"{MANUAL_GROUP} (0 electrodes)")
        for name, electrodes in groups.items():
            self.group_list.insert("end", f"{name} ({len(electrodes)} electrodes)")
        self.axis.clear()
        self.color_axis.clear()
        self.color_axis.set_axis_on()
        image = self.axis.imshow(rate_map, origin="lower")
        self.axis.set(title=bx_path.name, xlabel="Column", ylabel="Row")
        self.figure.colorbar(image, cax=self.color_axis, label="FP/min")
        self.companion_var.set(
            f"JSON: {json_path.name}\n"
            f"Waveforms + activity map: {bx_path.name}\n"
            f"Dataset: {self.well_var.get().strip()}/FpForms"
        )
        self._redraw_map()

    def _active_group_name(self) -> str | None:
        if not self.original_groups:
            return None
        index = self.group_list.index(tk.ACTIVE)
        names = [MANUAL_GROUP, *self.original_groups]
        return names[index] if 0 <= index < len(names) else None

    def _selected_group_names(self) -> list[str]:
        names = [MANUAL_GROUP, *self.original_groups]
        return [names[index] for index in self.group_list.curselection()]

    def _electrodes_for(self, name: str) -> set[int]:
        return self.manual_electrodes if name == MANUAL_GROUP else self.refined_groups[name]

    def _redraw_map(self):
        for patch in self.map_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        self.map_patches = []
        selected = self._selected_group_names()
        palette = ("#FFFFFF", "#00FFFF", "#FFFF00", "#FF66FF", "#00FF66", "#FF9933")
        for group_index, name in enumerate(selected):
            for electrode in self._electrodes_for(name):
                row, column = divmod(electrode, 64)
                patch = Rectangle(
                    (column - .5, row - .5), 1, 1, fill=False,
                    edgecolor=palette[group_index % len(palette)], linewidth=1.4,
                )
                self.axis.add_patch(patch)
                self.map_patches.append(patch)
        self.canvas.draw_idle()

    def _map_click(self, event):
        name = self._active_group_name()
        if name is None or event.inaxes != self.axis or event.button != 1 or event.xdata is None:
            return
        column, row = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= row < 64 and 0 <= column < 64):
            return
        electrode = row * 64 + column
        eligible = (
            set().union(*self.original_groups.values())
            if name == MANUAL_GROUP else self.original_groups[name]
        )
        if electrode not in eligible:
            self.status_var.set(
                f"Row {row}, column {column} → electrode ID {electrode}; "
                f"not present in the matching JSON."
            )
            return
        selected = self._electrodes_for(name)
        if electrode in selected:
            selected.remove(electrode)
        else:
            selected.add(electrode)
        index = [MANUAL_GROUP, *self.original_groups].index(name)
        self.group_list.delete(index)
        if name == MANUAL_GROUP:
            label = f"{name} ({len(selected)} electrodes)"
        else:
            label = f"{name} ({len(selected)}/{len(self.original_groups[name])} electrodes)"
        self.group_list.insert(index, label)
        self.group_list.selection_set(index)
        self.group_list.activate(index)
        state = "included" if electrode in selected else "excluded"
        self.status_var.set(
            f"Row {row}, column {column} → electrode ID {electrode}; "
            f"{self.fp_counts.get(electrode, 0)} matching FP waveform(s); {state}."
        )
        self._redraw_map()

    def restore_active_group(self):
        name = self._active_group_name()
        if name is None:
            return
        if name == MANUAL_GROUP:
            self.manual_electrodes.clear()
        else:
            self.refined_groups[name] = set(self.original_groups[name])
        selected = self._electrodes_for(name)
        index = [MANUAL_GROUP, *self.original_groups].index(name)
        self.group_list.delete(index)
        self.group_list.insert(index, f"{name} ({len(selected)} electrodes)")
        self.group_list.selection_set(index)
        self.group_list.activate(index)
        self._redraw_map()

    def start(self):
        try:
            formats = {key for key, var in self.format_vars.items() if var.get()}
            selected_files = [Path(self.files[index]) for index in self.file_list.curselection()]
            if not selected_files:
                raise ValueError("Select at least one BRW file in the list.")
            if not formats:
                raise ValueError("Select at least one output format.")
            selected_names = self._selected_group_names()
            if not selected_names:
                raise ValueError("Select at least one JSON group.")
            jobs = []
            for source in selected_files:
                json_path = matching_companion(source, (".json",))
                bx_path = matching_companion(source, (".brx", ".bxr"))
                if json_path is None or bx_path is None:
                    raise ValueError(f"Matching JSON and BRX/BXR are required beside {source.name}.")
                groups = read_json_groups(json_path)
                if self.current_file == source:
                    groups = {name: set(self._electrodes_for(name)) for name in selected_names}
                else:
                    chosen = {}
                    if MANUAL_GROUP in selected_names:
                        eligible = set().union(*groups.values())
                        chosen[MANUAL_GROUP] = set(self.manual_electrodes) & eligible
                    chosen.update({
                        name: set(groups[name])
                        for name in selected_names if name != MANUAL_GROUP and name in groups
                    })
                    groups = chosen
                    if not groups:
                        raise ValueError(
                            f"None of the selected group names are present in {json_path.name}."
                        )
                # The original waveform script reads Well_*/FpChIdxs and
                # Well_*/FpForms. Those datasets live in the analyzed BXR/BRX,
                # not in a generic raw BRW file.
                jobs.append((bx_path, groups))
        except Exception as exc:
            messagebox.showerror("Cannot extract waveforms", str(exc))
            return
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Extracting group waveforms…")
        threading.Thread(
            target=self._worker,
            args=(jobs, self.well_var.get().strip(), formats, Path(self.output_var.get()).expanduser()),
            daemon=True,
        ).start()

    def _worker(self, jobs, well, formats, output):
        errors, count = [], 0
        for source, groups in jobs:
            for name, electrodes in groups.items():
                if not electrodes:
                    errors.append(f"{source.name} — {name}: no electrodes selected")
                    continue
                try:
                    count += len(save_waveform_plot(source, output, well, name, electrodes, formats))
                except Exception as exc:
                    errors.append(f"{source.name} — {name}: {exc}")
        self.after(0, self._finish, count, errors)

    def _finish(self, count, errors):
        self.progress.stop()
        self.run_button.configure(state="normal")
        if errors:
            self.status_var.set(f"Created {count} plot(s); {len(errors)} group(s) failed.")
            messagebox.showwarning("Waveform extraction", "\n".join(errors))
        else:
            self.status_var.set(f"Completed: created {count} plot(s) in {self.output_var.get()}")
            messagebox.showinfo("Waveform extraction", self.status_var.get())
