from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from .analysis_engine import AnalysisConfig, discover_recordings, run_analysis
    from .fieldmap import FieldMapFrame
    from .waveform import WaveformFrame
except ImportError:
    from analysis_engine import AnalysisConfig, discover_recordings, run_analysis
    from fieldmap import FieldMapFrame
    from waveform import WaveformFrame


LFP_METRICS = {
    "frequency_distribution_stats": "Skewness, Kurtosis and Gini",
    "electrode_frequencies": "Electrode Frequencies summary",
    "frequency": "Frequency (FP/min)",
    "cv_iei": "CV of electrode IEIs",
    "active_electrodes": "Active electrodes",
    "amplitude": "Amplitude (pA)",
    "lfp_sne_ratio": "LFP/SNE ratio",
    "lfp_sne_kdl_ratio": "LFP/SNE-KDL ratio",
}
MUA_METRICS = {
    "frequency_distribution_stats": "Skewness, Kurtosis and Gini",
    "electrode_frequencies": "Electrode Frequencies summary",
    "frequency": "Frequency (Hz)",
    "cv_iei": "CV of electrode IEIs",
    "active_electrodes": "Active electrodes",
    "amplitude": "Amplitude (pA)",
    "mua_lfp_ratio": "MUA/LFP ratio",
}
SNE_METRICS = {
    "sne_frequency": "SNE frequency (events/min)",
    "sne_kdl_frequency": "SNE-KDL frequency (events/min)",
    "sne_duration": "SNE duration",
    "cv_iei": "SNE IEI CV",
    "fano": "SNE Fano factor",
    "psd_peaks": "SNE PSD peaks",
    "acf_peaks": "SNE autocorrelation peaks",
}
PLOTS = {
    "sync": "_Sync",
    "sync_kde": "_Sync_KDE",
    "frequency_distribution_comparison": "_Freq_Distribution_Comparison",
}
PROPAGATION_METRICS = {
    "duration": "Event duration",
    "total_distance": "Total propagation distance",
    "maximum_distance": "Maximum distance from origin",
    "total_speed": "Total propagation speed",
    "expansion_distance": "Expansion distance",
    "expansion_speed": "Expansion speed",
    "retraction_distance": "Retraction distance",
    "retraction_speed": "Retraction speed",
}
ROUTE_METRICS = {
    "average_occurrence": "Average occurrence % (excluding OUT)",
    "overall_occurrence": "Overall group occurrence % (excluding OUT)",
    "full_routes": "Full propagation-route frequency",
    "sub_routes": "Sub-route frequency",
    "signal_origin": "Signal-origin percentage by group",
    "signal_end": "Signal-ending percentage by group",
}
PROPAGATION_PLOTS = {
    "lfp_heatmap": "LFP 64 × 64 electrode heatmap",
    "cats": "Centre-of-Activity Trajectory",
    "weighted_cats": "Combined Weighted CATS",
}


class CheckGroup(ttk.Frame):
    def __init__(self, parent, choices):
        super().__init__(parent)
        self.vars, self.widgets = {}, {}
        for row, (key, label) in enumerate(choices.items()):
            var = tk.BooleanVar(value=True)
            widget = ttk.Checkbutton(self, text=label, variable=var)
            widget.grid(row=row, column=0, sticky="w", pady=2)
            self.vars[key], self.widgets[key] = var, widget

    def selected(self):
        return {key for key, var in self.vars.items() if var.get()}

    def enable(self, enabled: bool):
        for widget in self.widgets.values():
            widget.configure(state="normal" if enabled else "disabled")

    def set_item_enabled(self, key: str, enabled: bool):
        if key in self.widgets:
            self.widgets[key].configure(state="normal" if enabled else "disabled")


class MEAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MEA Analysis Workbench")
        self.geometry("1240x900")
        self.minsize(1040, 760)
        self.inputs: list[str] = []
        self.messages: queue.Queue = queue.Queue()
        self._build()
        self.after(100, self._drain_messages)

    def _build(self):
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        fieldmap_tab = ttk.Frame(notebook)
        analysis_tab = ttk.Frame(notebook)
        waveform_tab = ttk.Frame(notebook)
        notebook.add(fieldmap_tab, text="1 · Electrode group selection (JSON)")
        notebook.add(analysis_tab, text="2 · Analysis")
        notebook.add(waveform_tab, text="3 · Waveform extraction")
        FieldMapFrame(fieldmap_tab, on_export=self._fieldmap_exported).pack(fill="both", expand=True)
        WaveformFrame(waveform_tab).pack(fill="both", expand=True)

        shell = ttk.Frame(analysis_tab)
        shell.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(shell, highlightthickness=0)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        root = ttk.Frame(self.canvas, padding=16)
        self.canvas_window = self.canvas.create_window((0, 0), window=root, anchor="nw")
        root.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width),
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda _event: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda _event: self.canvas.yview_scroll(1, "units"))

        ttk.Label(root, text="MEA Analysis Workbench", style="Title.TLabel").pack(anchor="w")
        ttk.Label(root, text="Configurable LFP, MUA and synchronous network event (SNE) analysis.").pack(anchor="w", pady=(2, 12))

        top = ttk.Frame(root)
        top.pack(fill="x")
        input_box = ttk.LabelFrame(top, text="1 · Input and output", style="Section.TLabelframe", padding=10)
        input_box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.input_list = tk.Listbox(input_box, height=6, selectmode="extended")
        self.input_list.grid(row=0, column=0, rowspan=4, sticky="nsew")
        for row, (text, cmd) in enumerate((
            ("Add file(s)…", self.add_files), ("Add folder…", self.add_folder),
            ("Remove", self.remove_inputs), ("Clear", self.clear_inputs),
        )):
            ttk.Button(input_box, text=text, command=cmd).grid(row=row, column=1, sticky="ew", padx=(8, 0))
        input_box.columnconfigure(0, weight=1)
        self.recursive = tk.BooleanVar(value=True)
        ttk.Checkbutton(input_box, text="Search folders recursively", variable=self.recursive).grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.output_var = tk.StringVar(value=str(Path.home() / "Desktop" / "MEA_Analysis"))
        ttk.Entry(input_box, textvariable=self.output_var).grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(input_box, text="Output…", command=self.choose_output).grid(row=5, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        signal_box = ttk.LabelFrame(top, text="2 · Signals", style="Section.TLabelframe", padding=10)
        signal_box.pack(side="left", fill="y", padx=(0, 8))
        self.lfp, self.mua = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ttk.Checkbutton(signal_box, text="LFP / field potentials", variable=self.lfp, command=self._sync_availability).pack(anchor="w", pady=3)
        ttk.Checkbutton(signal_box, text="MUA / spikes", variable=self.mua, command=self._sync_availability).pack(anchor="w", pady=3)
        ttk.Separator(signal_box).pack(fill="x", pady=8)
        ttk.Label(signal_box, text="Well").pack(anchor="w")
        self.well = tk.StringVar(value="Well_A1")
        ttk.Entry(signal_box, textvariable=self.well, width=18).pack(anchor="w")
        ttk.Separator(signal_box).pack(fill="x", pady=8)
        self.generate_raster = tk.BooleanVar(value=True)
        ttk.Checkbutton(signal_box, text="Export raster", variable=self.generate_raster).pack(anchor="w")
        self.raster_format_vars = {}
        raster_formats = ttk.Frame(signal_box)
        raster_formats.pack(anchor="w")
        for column, (key, label) in enumerate((("png", ".png"), ("svg", ".svg"), ("emf.svg", ".emf.svg"))):
            var = tk.BooleanVar(value=True)
            self.raster_format_vars[key] = var
            ttk.Checkbutton(raster_formats, text=label, variable=var).grid(row=0, column=column, padx=(0, 5))

        aggregation_box = ttk.LabelFrame(top, text="Initial summary choices", style="Section.TLabelframe", padding=10)
        aggregation_box.pack(side="left", fill="y")
        ttk.Label(aggregation_box, text="Applied to frequency, amplitude,\nCV and derived ratios:").pack(anchor="w")
        self.aggregation_vars = {}
        for key, label in (("mean", "Mean"), ("geometric_mean", "Geometric mean"), ("median", "Median")):
            var = tk.BooleanVar(value=key != "median")
            self.aggregation_vars[key] = var
            ttk.Checkbutton(aggregation_box, text=label, variable=var).pack(anchor="w", pady=2)

        metric_area = ttk.LabelFrame(root, text="3 · Network Activity", style="Section.TLabelframe", padding=10)
        metric_area.pack(fill="x", pady=10)
        self.network_activity_enabled = tk.BooleanVar(value=True)
        self.network_toggle = ttk.Checkbutton(
            metric_area, text="Enable Network Activity analysis",
            variable=self.network_activity_enabled, command=self._sync_availability,
        )
        self.network_toggle.pack(anchor="w", pady=(0, 8))
        metric_columns = ttk.Frame(metric_area)
        metric_columns.pack(fill="x")
        self.lfp_box = ttk.LabelFrame(metric_columns, text="LFP parameters", padding=8)
        self.lfp_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.lfp_checks = CheckGroup(self.lfp_box, LFP_METRICS)
        self.lfp_checks.pack(anchor="w")
        self.mua_box = ttk.LabelFrame(metric_columns, text="MUA parameters", padding=8)
        self.mua_box.pack(side="left", fill="both", expand=True, padx=5)
        self.mua_checks = CheckGroup(self.mua_box, MUA_METRICS)
        self.mua_checks.pack(anchor="w")
        self.sne_box = ttk.LabelFrame(metric_columns, text="SNE parameters (LFP only)", padding=8)
        self.sne_box.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.sne_checks = CheckGroup(self.sne_box, SNE_METRICS)
        self.sne_checks.pack(anchor="w")

        lower = ttk.Frame(root)
        lower.pack(fill="both", expand=True)
        plot_box = ttk.LabelFrame(lower, text="4 · Plots to generate", style="Section.TLabelframe", padding=10)
        plot_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.plot_checks = CheckGroup(plot_box, PLOTS)
        self.plot_checks.pack(anchor="w")

        group_box = ttk.LabelFrame(lower, text="SNE electrode groups", style="Section.TLabelframe", padding=10)
        group_box.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(group_box, text="Select one or more groups. “All groups” runs each JSON group separately.").pack(anchor="w")
        self.group_list = tk.Listbox(group_box, height=9, selectmode="multiple", exportselection=False)
        self.group_list.pack(fill="both", expand=True, pady=6)
        self.group_list.insert("end", "All groups")
        self.group_list.selection_set(0)
        self.refresh_groups_button = ttk.Button(group_box, text="Refresh from selected inputs", command=self.refresh_groups)
        self.refresh_groups_button.pack(anchor="w")

        settings = ttk.LabelFrame(lower, text="5 · Detection settings", style="Section.TLabelframe", padding=10)
        settings.pack(side="left", fill="both", expand=True, padx=(5, 0))
        fields = [
            ("LFP active threshold (/min)", "lfp_threshold", "0.8"),
            ("MUA active threshold (Hz)", "mua_threshold", "0"),
            ("SNE minimum active (%)", "sne_percent", "40"),
            ("KDE bandwidth", "bandwidth", "0.01"),
            ("KDE boundary (%)", "boundary", "30"),
            ("Fano window (s)", "fano_window", "10"),
            ("ACF maximum lag", "acf_lag", "100"),
            ("Raster start (s)", "raster_start", "0"),
            ("Raster end (s)", "raster_end", "60"),
        ]
        self.fields = {}
        for row, (label, name, default) in enumerate(fields):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            self.fields[name] = var
            ttk.Entry(settings, textvariable=var, width=10).grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)

        propagation_area = ttk.LabelFrame(
            root, text="6 · Signal propagation analysis (LFP only)",
            style="Section.TLabelframe", padding=10,
        )
        propagation_area.pack(fill="x", pady=(10, 0))
        propagation_top = ttk.Frame(propagation_area)
        propagation_top.pack(fill="x", pady=(0, 8))
        self.propagation_enabled = tk.BooleanVar(value=False)
        self.propagation_toggle = ttk.Checkbutton(
            propagation_top, text="Enable signal propagation analysis",
            variable=self.propagation_enabled, command=self._sync_availability,
        )
        self.propagation_toggle.pack(side="left")
        ttk.Label(propagation_top, text="Regions:").pack(side="left", padx=(20, 4))
        self.propagation_region_vars = {}
        self.propagation_region_widgets = {}
        for key in ("HP", "CTX"):
            var = tk.BooleanVar(value=True)
            widget = ttk.Checkbutton(propagation_top, text=key, variable=var)
            widget.pack(side="left", padx=3)
            self.propagation_region_vars[key] = var
            self.propagation_region_widgets[key] = widget

        prop_metrics_box = ttk.LabelFrame(propagation_area, text="Propagation kinematics", padding=8)
        prop_metrics_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.propagation_checks = CheckGroup(prop_metrics_box, PROPAGATION_METRICS)
        self.propagation_checks.pack(anchor="w")

        route_box = ttk.LabelFrame(propagation_area, text="Propagation routes", padding=8)
        route_box.pack(side="left", fill="both", expand=True, padx=5)
        self.route_checks = CheckGroup(route_box, ROUTE_METRICS)
        self.route_checks.pack(anchor="w")

        prop_plots_box = ttk.LabelFrame(propagation_area, text="Propagation plots", padding=8)
        prop_plots_box.pack(side="left", fill="both", expand=True, padx=5)
        self.propagation_plot_checks = CheckGroup(prop_plots_box, PROPAGATION_PLOTS)
        self.propagation_plot_checks.pack(anchor="w")

        prop_settings = ttk.LabelFrame(propagation_area, text="Propagation settings", padding=8)
        prop_settings.pack(side="left", fill="both", expand=True, padx=(5, 0))
        propagation_fields = [
            ("Minimum active electrodes (%)", "prop_min_active", "40"),
            ("Histogram bins", "prop_bins", "200"),
            ("Event boundary (%)", "prop_boundary", "0"),
            ("Centroid time step (s)", "centroid_step", "0.01"),
            ("Centroid tolerance (s)", "centroid_tolerance", "0.025"),
            ("Electrode pitch (μm)", "electrode_pitch", "81"),
            ("Route groups (comma-separated)", "route_groups", "CA1,CA2,CA3,DG,SUBV"),
        ]
        for row, (label, name, default) in enumerate(propagation_fields):
            ttk.Label(prop_settings, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            self.fields[name] = var
            ttk.Entry(prop_settings, textvariable=var, width=22).grid(
                row=row, column=1, sticky="ew", padx=(8, 0), pady=2
            )

        bottom = ttk.Frame(root)
        bottom.pack(fill="both", expand=True, pady=(10, 0))
        buttons = ttk.Frame(bottom)
        buttons.pack(fill="x")
        self.run_button = ttk.Button(buttons, text="Run analysis", command=self.start)
        self.run_button.pack(side="left")
        ttk.Button(buttons, text="Save configuration…", command=self.save_configuration).pack(side="left", padx=6)
        ttk.Button(buttons, text="Open results", command=self.open_results).pack(side="left")
        self.progress = ttk.Progressbar(buttons, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(20, 0))
        self.log = tk.Text(bottom, height=8, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self._sync_availability()

    def _fieldmap_exported(self, bxr_path: str, _json_path: str):
        if bxr_path and bxr_path not in self.inputs:
            self.inputs.append(bxr_path)
            self.input_list.insert("end", bxr_path)
        self.refresh_groups()

    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def add_files(self):
        self._add(filedialog.askopenfilenames(filetypes=[("MEA recordings", "*.brx *.bxr"), ("All files", "*.*")]))

    def add_folder(self):
        path = filedialog.askdirectory()
        if path:
            self._add([path])

    def _add(self, paths):
        for path in paths:
            if path not in self.inputs:
                self.inputs.append(path)
                self.input_list.insert("end", path)
        self.refresh_groups()

    def remove_inputs(self):
        for index in reversed(self.input_list.curselection()):
            self.inputs.pop(index)
            self.input_list.delete(index)
        self.refresh_groups()

    def clear_inputs(self):
        self.inputs.clear()
        self.input_list.delete(0, "end")
        self.refresh_groups()

    def choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def refresh_groups(self):
        groups = set()
        for recording in discover_recordings(self.inputs, self.recursive.get()):
            json_path = recording.with_suffix(".json")
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                groups.update(str(g["UserDefinedName"]) for g in payload.get("Groups", []))
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        self.group_list.delete(0, "end")
        for name in ["All groups", "Full", *sorted(groups)]:
            self.group_list.insert("end", name)
        self.group_list.selection_set(0)

    def _sync_availability(self):
        lfp, mua = self.lfp.get(), self.mua.get()
        network_active = self.network_activity_enabled.get()
        self.lfp_checks.enable(lfp and network_active)
        self.mua_checks.enable(mua and network_active)
        self.sne_checks.enable(lfp and network_active)
        self.plot_checks.enable(network_active and (lfp or mua))
        self.plot_checks.set_item_enabled("sync", lfp and network_active)
        self.plot_checks.set_item_enabled("sync_kde", lfp and network_active)
        self.mua_checks.set_item_enabled("mua_lfp_ratio", lfp and mua and network_active)
        self.lfp_checks.set_item_enabled("lfp_sne_ratio", lfp and network_active)
        self.group_list.configure(state="normal" if lfp and network_active else "disabled")
        self.refresh_groups_button.configure(state="normal" if lfp and network_active else "disabled")
        self.propagation_toggle.configure(state="normal" if lfp else "disabled")
        propagation_active = lfp and self.propagation_enabled.get()
        self.propagation_checks.enable(propagation_active)
        self.route_checks.enable(propagation_active)
        self.propagation_plot_checks.enable(propagation_active)
        for widget in self.propagation_region_widgets.values():
            widget.configure(state="normal" if propagation_active else "disabled")

    def config(self):
        if not self.inputs:
            raise ValueError("Add at least one recording or folder.")
        if not self.lfp.get() and not self.mua.get():
            raise ValueError("Select LFP, MUA, or both.")
        aggregations = {key for key, var in self.aggregation_vars.items() if var.get()}
        if not aggregations:
            raise ValueError("Select at least one initial summary choice: mean, geometric mean, or median.")
        selections = self.group_list.curselection()
        sne_groups = [self.group_list.get(i) for i in selections] or ["All groups"]
        raster_formats = {key for key, var in self.raster_format_vars.items() if var.get()}
        if self.generate_raster.get() and not raster_formats:
            raise ValueError("Select at least one raster output format.")
        propagation_regions = {
            key for key, var in self.propagation_region_vars.items() if var.get()
        }
        propagation_enabled = self.propagation_enabled.get() and self.lfp.get()
        network_activity_enabled = self.network_activity_enabled.get()
        if not network_activity_enabled and not propagation_enabled:
            raise ValueError("Enable Network Activity, Signal Propagation, or both.")
        if propagation_enabled and not propagation_regions:
            raise ValueError("Select HP, CTX, or both for propagation analysis.")
        route_group_names = [
            value.strip() for value in self.fields["route_groups"].get().split(",") if value.strip()
        ]
        if propagation_enabled and self.route_checks.selected() and not route_group_names:
            raise ValueError("Enter at least one route group.")
        prop_min_active = float(self.fields["prop_min_active"].get())
        prop_boundary = float(self.fields["prop_boundary"].get())
        prop_bins = int(self.fields["prop_bins"].get())
        centroid_step = float(self.fields["centroid_step"].get())
        centroid_tolerance = float(self.fields["centroid_tolerance"].get())
        electrode_pitch = float(self.fields["electrode_pitch"].get())
        if propagation_enabled:
            if not 0 <= prop_min_active <= 100 or not 0 <= prop_boundary <= 100:
                raise ValueError("Propagation percentages must be between 0 and 100.")
            if prop_bins < 2:
                raise ValueError("Propagation histogram bins must be at least 2.")
            if centroid_step <= 0 or centroid_tolerance < 0 or electrode_pitch <= 0:
                raise ValueError("Centroid step and electrode pitch must be positive; tolerance cannot be negative.")
        return AnalysisConfig(
            inputs=self.inputs.copy(), output_dir=self.output_var.get(), recursive=self.recursive.get(),
            well=self.well.get().strip(), analyze_lfp=self.lfp.get(), analyze_mua=self.mua.get(),
            aggregations=aggregations, lfp_metrics=self.lfp_checks.selected(),
            mua_metrics=self.mua_checks.selected(), sne_metrics=self.sne_checks.selected(),
            sne_groups=sne_groups, generate_raster=self.generate_raster.get(),
            raster_formats=raster_formats,
            active_threshold_lfp=float(self.fields["lfp_threshold"].get()),
            active_threshold_mua=float(self.fields["mua_threshold"].get()),
            sne_min_active_percent=float(self.fields["sne_percent"].get()),
            kde_bandwidth=float(self.fields["bandwidth"].get()),
            boundary_percent=float(self.fields["boundary"].get()),
            fano_window_s=float(self.fields["fano_window"].get()),
            acf_max_lag=int(self.fields["acf_lag"].get()),
            raster_start_s=float(self.fields["raster_start"].get()),
            raster_end_s=float(self.fields["raster_end"].get()),
            plots=self.plot_checks.selected(),
            propagation_enabled=propagation_enabled,
            propagation_regions=propagation_regions,
            propagation_metrics=self.propagation_checks.selected(),
            route_metrics=self.route_checks.selected(),
            propagation_plots=self.propagation_plot_checks.selected(),
            propagation_min_active_percent=prop_min_active,
            propagation_bins=prop_bins,
            propagation_boundary_percent=prop_boundary,
            centroid_step_s=centroid_step,
            centroid_tolerance_s=centroid_tolerance,
            electrode_pitch_um=electrode_pitch,
            route_group_names=route_group_names,
            network_activity_enabled=network_activity_enabled,
        )

    def save_configuration(self):
        try:
            config = self.config()
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                raw = vars(config).copy()
                for key in (
                    "aggregations", "lfp_metrics", "mua_metrics", "sne_metrics", "plots",
                    "raster_formats", "propagation_regions", "propagation_metrics",
                    "route_metrics", "propagation_plots",
                ):
                    raw[key] = sorted(raw[key])
                Path(path).write_text(json.dumps(raw, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))

    def start(self):
        try:
            config = self.config()
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self._write_log("Starting analysis…")
        threading.Thread(target=self._worker, args=(config,), daemon=True).start()

    def _worker(self, config):
        try:
            run_analysis(config, lambda line: self.messages.put(("log", line)))
            self.messages.put(("done", "Analysis completed."))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def _drain_messages(self):
        try:
            while True:
                kind, text = self.messages.get_nowait()
                if kind == "log":
                    self._write_log(text)
                else:
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    self._write_log(text)
                    (messagebox.showinfo if kind == "done" else messagebox.showerror)("MEA Analysis", text)
        except queue.Empty:
            pass
        self.after(100, self._drain_messages)

    def _write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", str(text) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_results(self):
        path = Path(self.output_var.get()).expanduser()
        if not path.exists():
            messagebox.showwarning("Results", "The output folder does not exist yet.")
            return
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


if __name__ == "__main__":
    MEAApp().mainloop()
