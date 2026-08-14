# MEA Analysis Workbench

A three-tab desktop GUI for electrode-group creation, configurable batch analysis
of `.brx`/`.bxr` recordings, and FP waveform extraction from `.brw` files.
It was separated from the original research scripts so paths, detection thresholds,
metrics, and plots can be selected without editing Python source code.
The application opens on **Electrode group selection (JSON)**. The second tab is
the vertically scrollable analysis interface, and the third is **Waveform extraction**.
`run_gui.py` resolves all project imports from its own folder, so it can be run
directly in Spyder without executing another analysis script first.

## Input convention

Each recording must have a group-definition JSON file with the same basename:

```text
recording_01.bxr
recording_01.json
```

The JSON must contain `Groups`, with `UserDefinedName` and `PixelIndexes`, as in
the existing scripts. Folders can be searched recursively. `Well_A1` is the
default well and can be changed in the GUI.

## Electrode group selection

The first tab accepts multiple BXR/BRX files and immediately loads the first
FP-rate map. It supports threshold filtering, click or freehand-lasso
add/remove selection, multiple named and coloured groups, group deletion, and
BrainWave-compatible JSON export. Electrode groups can be drawn directly on the
FP-rate map or, optionally, after loading a corresponding slice image. For an
image overlay, **Manually fit image to FP-map boundaries** asks the user to
click four points in the slice image—lower-left, lower-right, upper-right, and
upper-left—that correspond to the FP-map limits. A projective transformation
then maps that quadrilateral to the outer boundaries of the 64 × 64 FP-rate map
and shows the slice as an adjustable transparent overlay. This visual
registration never changes the zero-based electrode IDs written to the
BrainWave JSON file.

When an aligned slice image is present, export also writes a same-basename
`_image_registration.json` sidecar containing the source-image path, four
manually selected boundary vertices, projective transform, and overlay opacity.
The normal group file remains
the same-basename `.json` beside the current recording, and the tab advances to
the next selected file. The recording is also added to the Analysis tab
automatically. A **Skip current file** button advances without creating or
replacing a JSON file.

The FP-rate heatmap and its colorbar use fixed, separate plotting areas, so the
map retains the same size while advancing through a multi-file queue.

## Waveform extraction

The third tab accepts one or more `.brw` files as recording identifiers and
automatically finds the same-basename JSON and BRX/BXR files beside each
recording. Following the original waveform script, `FpChIdxs` and `FpForms` are
read from `Well_*/` in the matching analyzed BRX/BXR—not from the raw BRW.
The BRX/BXR LFP-rate map is displayed with the chosen JSON groups overlaid.
Clicking the map reports the zero-based row, column, electrode ID, and matching
FP-waveform count. Electrodes can be
included or excluded by clicking the map, but only electrodes belonging to the
active JSON group are eligible. A separate **Manual Selection** entry permits
individual map selection from the union of all electrodes present in the JSON.
Each selected group produces its own average FP
waveform and ±1 standard-deviation plot in `.png`, `.svg`, and/or `.emf.svg`.

## Run on Windows

The easiest method is to double-click `install_and_run.bat`. It creates a local
Python environment, installs the required packages, and starts the application.

For manual setup:

1. Install 64-bit Python 3.10 or newer from python.org. During installation,
   enable **Add Python to PATH**.
2. Open Command Prompt in this folder.
3. Create and activate an environment:

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   py -m pip install -r requirements.txt
   ```

4. Start the GUI:

   ```bat
   py run_gui.py
   ```

Tkinter is included in the standard Windows Python installer.

## Available analysis

- a default-enabled **Network Activity** module that can be switched off when
  running signal-propagation analysis alone
- independent Mean, Geometric mean, and Median summaries for frequency,
  amplitude, electrode IEI CV, LFP/SNE, and MUA/LFP ratios
- optional `LFP_Summary_Electrode_Frequencies.xlsx` and
  `MUA_Summary_Electrode_Frequencies.xlsx`, with one worksheet per group
- separate, signal-aware LFP, MUA, and SNE parameter panels
- LFP/field-potential and MUA/spike event rates
- active-electrode count and mean waveform amplitude when waveforms exist
- mean and geometric-mean rates
- direct SNE and KDE-based SNE-KDL detection for one, several, or all electrode groups
- SNE/SNE-KDL frequency, LFP/SNE and LFP/SNE-KDL ratios, duration, IEI CV,
  Fano factor, PSD and ACF peaks
- firing-rate skewness, kurtosis and Gini coefficient
- exact selectable plot families from the original workflow:
  `_Sync`, `_Sync_KDE`, and `_Freq_Distribution_Comparison`

The output contains a folder per recording, a per-recording Excel workbook,
`Summary_Results.xlsx`, the exact `analysis_config.json`, plots, and an error log
when individual files fail.

`Summary_Results.xlsx` uses separate `LFP`, `SNE`, and `MUA` worksheets. Each
worksheet contains only columns relevant to that signal, with frozen headers,
filters, readable widths, and consistent numeric formatting.

SNE controls require LFP. MUA/LFP requires both LFP and MUA. When **All groups**
is selected, SNE detection runs for `Full` and for every electrode group in the
recording's matching JSON file.

## Signal propagation analysis

The optional LFP propagation panel uses a chronological centroid trajectory:
every centroid remains paired with its sampling time. It does not infer movement
order from a 64 × 64 occupancy matrix.

Selectable event-level measurements for HP and CTX include duration, total
distance and speed, maximum distance from origin, expansion distance and speed,
and retraction distance and speed. Both region worksheets end with a nonzero
average row.

Selectable route summaries exclude `OUT` and include average and overall group
occurrence, full routes, sub-routes, signal origins, and signal endings. Route
group names are configurable and default to `CA1, CA2, CA3, DG, SUBV`.

Propagation plots include the LFP 64 × 64 frequency heatmap, chronological
Centre-of-Activity Trajectories, and Combined Weighted CATS. Detection controls
include active-electrode percentage, histogram bins, event-boundary percentage,
centroid time step/tolerance, and electrode pitch in μm.

The main output folder also contains `Signal Propagation Analysis summary.xlsx`.
Its HP and CTX propagation sheets contain one nonzero-average row per recording;
the route sheets cumulatively combine results from every analyzed recording with
a `File` column. For readability, every cumulative sheet except `HP_Propagation`
contains one blank separator row between recordings. Per-recording propagation
workbooks remain inside their recording folders.

## Notes

- The original scripts are not modified.
- The GUI supports both `.brx` and `.bxr` because both extensions are used in
  MEA workflows and the supplied scripts use `.bxr`.
- Missing MUA datasets are reported and skipped without preventing LFP analysis.
- The amplitude calculation is only available when the corresponding waveform
  dataset and wavelength metadata are present.
