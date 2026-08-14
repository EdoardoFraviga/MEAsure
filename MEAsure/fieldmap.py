from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import h5py
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates


GRID_SIDE = 64
ELECTRODES_PER_WELL = GRID_SIDE * GRID_SIDE


def load_fp_rate_map(path: str | Path, well: str) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        sampling_rate = float(np.asarray(handle.attrs["SamplingRate"]).reshape(-1)[0])
        toc = np.asarray(handle["TOC"])
        if not toc.size:
            raise ValueError("TOC is empty; recording duration cannot be calculated.")
        duration_min = float(toc.reshape(-1, toc.shape[-1])[-1, -1]) / sampling_rate / 60
        if duration_min <= 0:
            raise ValueError("Recording duration is zero.")
        if well not in handle or "FpChIdxs" not in handle[well]:
            raise KeyError(f"'{well}/FpChIdxs' was not found.")
        channels = np.asarray(handle[well]["FpChIdxs"], dtype=np.int64).reshape(-1)
    local = channels % ELECTRODES_PER_WELL
    return (np.bincount(local, minlength=ELECTRODES_PER_WELL) / duration_min).reshape(64, 64), duration_min


def export_groups_json(path: str | Path, groups: list[dict]) -> None:
    exported = []
    for index, group in enumerate(groups):
        rows, columns = np.where(group["mask"])
        red, green, blue = (int(value) for value in group["color"])
        exported.append({
            "$type": "_3Brain.Common.MsaPixelsGroup, 3Brain.Common",
            "PixelIndexes": [int(row * 64 + column) for row, column in zip(rows, columns)],
            "GroupCriteria": None,
            "Name": f"Unit Group {index + 1}",
            "UserDefinedName": str(group["name"]),
            "Color": f"{red}, {green}, {blue}",
            "Visible": True,
            "Index": -1,
        })
    payload = {"Model": 0, "ChipRoi": 0, "Groups": exported, "JsonVersion": 0}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def solve_affine_transform(image_points: list[tuple[float, float]],
                           electrode_points: list[tuple[float, float]]) -> np.ndarray:
    """Return a Matplotlib affine matrix mapping image pixels to map coordinates."""
    if len(image_points) != len(electrode_points) or len(image_points) < 3:
        raise ValueError("Choose at least three matching landmark pairs.")
    source = np.asarray(image_points, dtype=float)
    target = np.asarray(electrode_points, dtype=float)
    design = np.column_stack([source, np.ones(len(source))])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("The slice-image landmarks must not be collinear.")
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("The selected landmarks do not define a valid alignment.")
    return np.array([
        [coefficients[0, 0], coefficients[1, 0], coefficients[2, 0]],
        [coefficients[0, 1], coefficients[1, 1], coefficients[2, 1]],
        [0.0, 0.0, 1.0],
    ])


def fp_map_boundary_vertices() -> list[tuple[float, float]]:
    """Return FP-map corners clockwise: lower-left, lower-right, upper-right, upper-left."""
    return [(-0.5, -0.5), (63.5, -0.5), (63.5, 63.5), (-0.5, 63.5)]


def solve_homography(image_points: list[tuple[float, float]],
                     electrode_points: list[tuple[float, float]]) -> np.ndarray:
    """Return a projective transform mapping four image points to FP-map points."""
    if len(image_points) != 4 or len(electrode_points) != 4:
        raise ValueError("Select exactly four slice-image boundary vertices.")
    rows, target = [], []
    for (x, y), (u, v) in zip(image_points, electrode_points):
        rows.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
        target.extend([u, v])
    design = np.asarray(rows, dtype=float)
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("The four slice-image vertices must form a valid quadrilateral.")
    coefficients = np.linalg.solve(design, np.asarray(target, dtype=float))
    return np.array([
        [coefficients[0], coefficients[1], coefficients[2]],
        [coefficients[3], coefficients[4], coefficients[5]],
        [coefficients[6], coefficients[7], 1.0],
    ])


def warp_image_to_fp_map(image: np.ndarray, homography: np.ndarray, output_size: int = 512) -> np.ndarray:
    """Perspective-warp a selected slice-image quadrilateral to the 64 × 64 map."""
    source = np.asarray(image)
    if source.ndim == 2:
        source = source[..., np.newaxis]
    if source.shape[-1] not in (1, 3, 4):
        raise ValueError("The slice image must be grayscale, RGB, or RGBA.")
    source = source.astype(float)
    if np.issubdtype(image.dtype, np.integer):
        source /= np.iinfo(image.dtype).max
    elif source.max(initial=0) > 1:
        source /= source.max()
    height, width = source.shape[:2]
    x_values = np.linspace(-0.5, 63.5, output_size)
    y_values = np.linspace(-0.5, 63.5, output_size)
    target_x, target_y = np.meshgrid(x_values, y_values)
    target = np.vstack([target_x.ravel(), target_y.ravel(), np.ones(target_x.size)])
    inverse = np.linalg.inv(homography)
    mapped = inverse @ target
    source_x = mapped[0] / mapped[2]
    source_y = mapped[1] / mapped[2]
    valid = (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)
    coordinates = np.vstack([source_y, source_x])
    if source.shape[-1] == 1:
        rgb = np.repeat(source, 3, axis=-1)
        alpha_source = np.ones((height, width), dtype=float)
    else:
        rgb = source[..., :3]
        alpha_source = source[..., 3] if source.shape[-1] == 4 else np.ones((height, width), dtype=float)
    warped = np.empty((output_size, output_size, 4), dtype=float)
    for channel in range(3):
        warped[..., channel] = map_coordinates(rgb[..., channel], coordinates, order=1, mode="constant", cval=0).reshape(output_size, output_size)
    warped[..., 3] = map_coordinates(alpha_source, coordinates, order=1, mode="constant", cval=0).reshape(output_size, output_size)
    warped[..., 3] *= valid.reshape(output_size, output_size)
    return np.clip(warped, 0, 1)


def export_image_registration(path: str | Path, source_image: str | Path,
                              image_points: list[tuple[float, float]],
                              electrode_points: list[tuple[float, float]],
                              transform: np.ndarray, opacity: float) -> None:
    """Write the optional, reproducible image-overlay metadata beside group JSON."""
    payload = {
        "SourceImage": str(Path(source_image)),
        "ImageLandmarks": [[float(x), float(y)] for x, y in image_points],
        "ElectrodeLandmarks": [[float(x), float(y)] for x, y in electrode_points],
        "ImageToElectrodeHomography": np.asarray(transform, dtype=float).round(12).tolist(),
        "AlignmentMethod": "four manually selected slice-image vertices mapped to 64x64 FP-map boundaries",
        "Opacity": float(opacity),
        "CoordinateSystem": "zero-based 64x64 electrode map; x=column, y=row",
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FieldMapFrame(ttk.Frame):
    def __init__(self, parent, on_export=None):
        super().__init__(parent, padding=16)
        self.on_export = on_export
        self.paths: list[str] = []
        self.current_index = -1
        self.rate_map: np.ndarray | None = None
        self.manual_mask = np.zeros((64, 64), dtype=bool)
        self.threshold_mask = np.ones((64, 64), dtype=bool)
        self.groups: list[dict] = []
        self.selection_patches, self.group_patches = [], []
        self.drawing, self.points, self.line = False, [], None
        self.slice_image: np.ndarray | None = None
        self.slice_image_path = ""
        self.image_landmarks: list[tuple[float, float]] = []
        self.electrode_landmarks: list[tuple[float, float]] = []
        self.registration_matrix: np.ndarray | None = None
        self.warped_overlay: np.ndarray | None = None
        self.overlay_artist = None
        self._build()

    def _build(self):
        ttk.Label(self, text="Electrode group selection (JSON)", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text="Load a BXR file and select electrodes directly on the 64 × 64 FP-rate map, or align a corresponding slice image as an optional overlay.",
        ).pack(anchor="w", pady=(2, 12))

        top = ttk.LabelFrame(self, text="BXR file and settings", style="Section.TLabelframe", padding=10)
        top.pack(fill="x")
        self.path_var, self.well_var = tk.StringVar(), tk.StringVar(value="Well_A1")
        self.threshold_var, self.mode_var = tk.StringVar(value="0.8"), tk.StringVar(value="add")
        self.queue_var = tk.StringVar(value="No files selected")
        ttk.Label(top, text="Current BXR/BRX").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.path_var, state="readonly").grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Select file(s)…", command=self.choose_files).grid(row=0, column=2)
        ttk.Label(top, textvariable=self.queue_var).grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(top, text="Well").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.well_var, width=15).grid(row=2, column=1, sticky="w", padx=6, pady=(8, 0))
        top.columnconfigure(1, weight=1)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=10)
        plot_box = ttk.LabelFrame(body, text="FP-rate map (64 × 64 electrode coordinates)", padding=6)
        plot_box.pack(side="left", fill="both", expand=True, padx=(0, 6))
        # Keep the heatmap and colorbar in fixed axes.  Creating a new
        # colorbar with ``ax=self.axis`` repeatedly shrinks the map axis.
        self.figure = Figure(figsize=(7, 6))
        self.axis = self.figure.add_axes((0.09, 0.10, 0.74, 0.82))
        self.color_axis = self.figure.add_axes((0.87, 0.10, 0.04, 0.82))
        self.color_axis.set_axis_off()
        self.axis.set(title="Load a BXR/BRX file to begin", xlabel="Electrode column (0–63)", ylabel="Electrode row (0–63)")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_release_event", self._release)

        side = ttk.LabelFrame(body, text="Selection and groups", padding=10)
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Threshold (FP/min)").grid(row=0, column=0, sticky="w")
        ttk.Entry(side, textvariable=self.threshold_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Button(side, text="Apply", command=self.apply_threshold).grid(row=0, column=2)
        ttk.Label(side, text="Lasso/click mode").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Radiobutton(side, text="Add", value="add", variable=self.mode_var).grid(row=1, column=1, pady=(10, 0))
        ttk.Radiobutton(side, text="Remove", value="remove", variable=self.mode_var).grid(row=1, column=2, pady=(10, 0))
        ttk.Button(side, text="Clear current selection", command=self.clear_selection).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Separator(side).grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(side, text="Optional slice-image overlay", font=("TkDefaultFont", 9, "bold")).grid(
            row=4, column=0, columnspan=3, sticky="w"
        )
        ttk.Button(side, text="Load slice image…", command=self.load_slice_image).grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )
        self.register_button = ttk.Button(side, text="Manually fit image to FP-map boundaries…", command=self.open_boundary_registration)
        self.register_button.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.register_button.configure(state="disabled")
        self.show_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Show aligned image", variable=self.show_overlay_var,
                        command=self._draw_overlay).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(side, text="Image opacity").grid(row=8, column=0, sticky="w", pady=(4, 0))
        # Make a successfully aligned slice image clearly visible by default.
        # Individual FP-map activity is still visible beneath it and the user can
        # lower the opacity at any point.
        self.overlay_opacity_var = tk.IntVar(value=70)
        ttk.Scale(side, from_=0, to=100, variable=self.overlay_opacity_var,
                  command=lambda _value: self._draw_overlay()).grid(row=8, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=(4, 0))
        ttk.Button(side, text="Remove slice image", command=self.clear_slice_image).grid(
            row=9, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )
        ttk.Separator(side).grid(row=10, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(side, text="Created groups").grid(row=11, column=0, columnspan=3, sticky="w")
        self.group_list = tk.Listbox(side, width=34, height=14, exportselection=False)
        self.group_list.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=5)
        ttk.Button(side, text="Add group from selection", command=self.add_group).grid(
            row=13, column=0, columnspan=3, sticky="ew"
        )
        ttk.Button(side, text="Delete selected group", command=self.delete_group).grid(
            row=14, column=0, columnspan=3, sticky="ew", pady=5
        )
        ttk.Button(side, text="Export JSON and continue", command=self.export_json).grid(
            row=15, column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )
        ttk.Button(side, text="Skip current file", command=self.skip_current).grid(
            row=16, column=0, columnspan=3, sticky="ew", pady=(5, 0)
        )
        side.rowconfigure(12, weight=1)
        self.status_var = tk.StringVar(value="No BXR file loaded.")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w")

    def choose_files(self):
        paths = list(filedialog.askopenfilenames(
            filetypes=[("3Brain BXR/BRX", "*.bxr *.brx"), ("All files", "*.*")]
        ))
        if paths:
            self.paths = paths
            self.current_index = 0
            self._load_current()

    def _load_current(self):
        if not (0 <= self.current_index < len(self.paths)):
            return
        self.path_var.set(self.paths[self.current_index])
        self.queue_var.set(f"File {self.current_index + 1} of {len(self.paths)}")
        try:
            rate_map, duration = load_fp_rate_map(self.path_var.get(), self.well_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Cannot load BXR", str(exc))
            self._advance()
            return
        self.rate_map = rate_map
        self.manual_mask[:] = False
        self.threshold_mask[:] = True
        self.groups.clear()
        self.group_list.delete(0, "end")
        self.clear_slice_image(redraw=False)
        self.axis.clear()
        self.color_axis.clear()
        self.color_axis.set_axis_on()
        image = self.axis.imshow(rate_map, origin="lower", zorder=0)
        self.axis.set(title=Path(self.path_var.get()).name, xlabel="Electrode column (0–63)", ylabel="Electrode row (0–63)")
        self.axis.set_xlim(-.5, 63.5)
        self.axis.set_ylim(-.5, 63.5)
        self.figure.colorbar(image, cax=self.color_axis, label="FP rate (events/min)")
        self.status_var.set(f"Loaded {Path(self.path_var.get()).name} ({duration:.2f} min).")
        self.canvas.draw_idle()

    def _advance(self):
        self.current_index += 1
        if self.current_index < len(self.paths):
            self._load_current()
        else:
            self.path_var.set("")
            self.queue_var.set(f"Completed {len(self.paths)} file(s)")
            self.status_var.set("All selected BXR/BRX files have been processed.")
            self.rate_map = None
            self.groups.clear()
            self.group_list.delete(0, "end")
            self.clear_slice_image(redraw=False)

    def skip_current(self):
        if self.rate_map is None:
            return
        source = Path(self.path_var.get())
        if messagebox.askyesno(
            "Skip recording?", f"Skip {source.name} without exporting a JSON file?"
        ):
            self._advance()

    def apply_threshold(self):
        if self.rate_map is None:
            messagebox.showinfo("FP map", "Load a BXR file first.")
            return
        try:
            self.threshold_mask = self.rate_map >= float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Threshold", "Enter a numeric threshold.")
            return
        self._redraw()

    def clear_selection(self):
        self.manual_mask[:] = False
        self._redraw()

    def load_slice_image(self):
        if self.rate_map is None:
            messagebox.showinfo("Slice image", "Load a BXR/BRX file first.")
            return
        path = filedialog.askopenfilename(
            title="Select the corresponding slice image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            image = np.asarray(imread(path))
            if image.ndim not in (2, 3):
                raise ValueError("The selected file is not a supported grayscale or colour image.")
        except Exception as exc:
            messagebox.showerror("Slice image", f"Could not load the image:\n{exc}")
            return
        self.slice_image = image
        self.slice_image_path = path
        self.image_landmarks, self.electrode_landmarks = [], []
        self.registration_matrix = None
        self.warped_overlay = None
        self.register_button.configure(state="normal")
        self._draw_overlay()
        self.status_var.set(
            f"Loaded slice image {Path(path).name}. Choose ‘Manually fit image to FP-map boundaries…’ to select its four FP-map boundary vertices."
        )

    def clear_slice_image(self, redraw=True):
        self.slice_image = None
        self.slice_image_path = ""
        self.image_landmarks, self.electrode_landmarks = [], []
        self.registration_matrix = None
        self.warped_overlay = None
        self.register_button.configure(state="disabled")
        if self.overlay_artist is not None:
            try:
                self.overlay_artist.remove()
            except (ValueError, AttributeError):
                pass
        self.overlay_artist = None
        if redraw:
            self.canvas.draw_idle()

    def open_boundary_registration(self):
        if self.rate_map is None or self.slice_image is None:
            return
        BoundaryRegistrationDialog(self, self.slice_image, self.rate_map, self._apply_boundary_registration)

    def _apply_boundary_registration(self, image_points, electrode_points, matrix):
        try:
            warped = warp_image_to_fp_map(self.slice_image, matrix)
        except Exception as exc:
            messagebox.showerror("Slice image alignment", str(exc), parent=self)
            return
        self.image_landmarks = image_points
        self.electrode_landmarks = electrode_points
        self.registration_matrix = matrix
        self.warped_overlay = warped
        self.show_overlay_var.set(True)
        self._draw_overlay()
        self.status_var.set(
            f"Mapped four manually selected vertices of {Path(self.slice_image_path).name} to the FP-map boundaries. Select electrodes normally on the overlaid map."
        )

    def _draw_overlay(self):
        if self.overlay_artist is not None:
            try:
                self.overlay_artist.remove()
            except (ValueError, AttributeError):
                pass
        self.overlay_artist = None
        if (self.rate_map is not None and self.warped_overlay is not None
                and self.show_overlay_var.get()):
            # Keep the tissue image above the FP-rate heatmap.  Electrode/group
            # outlines are assigned still higher z-orders in _redraw so manual
            # selection remains readable and usable through the overlay.
            self.overlay_artist = self.axis.imshow(
                self.warped_overlay, origin="lower", extent=(-.5, 63.5, -.5, 63.5),
                alpha=float(self.overlay_opacity_var.get()) / 100, interpolation="bilinear", zorder=10,
            )
            self.axis.set_xlim(-.5, 63.5)
            self.axis.set_ylim(-.5, 63.5)
        self._redraw()

    def _selected(self):
        return self.manual_mask & self.threshold_mask

    def _redraw(self):
        for patch in self.selection_patches + self.group_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        self.selection_patches, self.group_patches = [], []
        for row, column in zip(*np.where(self._selected())):
            patch = Rectangle(
                (column - .5, row - .5), 1, 1, fill=False,
                edgecolor="white", linewidth=1.2, zorder=20,
            )
            self.axis.add_patch(patch)
            self.selection_patches.append(patch)
        for group in self.groups:
            color = tuple(value / 255 for value in group["color"])
            for row, column in zip(*np.where(group["mask"])):
                patch = Rectangle(
                    (column - .5, row - .5), 1, 1, fill=False,
                    edgecolor=color, linewidth=1.3, zorder=21,
                )
                self.axis.add_patch(patch)
                self.group_patches.append(patch)
        self.canvas.draw_idle()

    def _press(self, event):
        if self.rate_map is None or event.inaxes != self.axis or event.button != 1 or event.xdata is None:
            return
        self.drawing, self.points = True, [(event.xdata, event.ydata)]
        (self.line,) = self.axis.plot(
            [event.xdata], [event.ydata], color="white", linewidth=1, zorder=30,
        )

    def _motion(self, event):
        if not self.drawing or event.inaxes != self.axis or event.xdata is None:
            return
        self.points.append((event.xdata, event.ydata))
        self.line.set_data([point[0] for point in self.points], [point[1] for point in self.points])
        self.canvas.draw_idle()

    def _release(self, event):
        if not self.drawing or event.button != 1:
            return
        self.drawing = False
        if self.line is not None:
            self.line.remove()
            self.line = None
        xs, ys = [p[0] for p in self.points], [p[1] for p in self.points]
        if len(self.points) < 3 or (max(xs) - min(xs) <= .5 and max(ys) - min(ys) <= .5):
            column, row = int(round(xs[-1])), int(round(ys[-1]))
            if 0 <= row < 64 and 0 <= column < 64:
                self.manual_mask[row, column] = self.mode_var.get() == "add"
        else:
            columns, rows = np.meshgrid(np.arange(64), np.arange(64))
            inside = MplPath(self.points + [self.points[0]]).contains_points(
                np.column_stack([columns.ravel(), rows.ravel()])
            ).reshape(64, 64)
            self.manual_mask[inside] = self.mode_var.get() == "add"
        self.points = []
        self._redraw()

    def add_group(self):
        mask = self._selected().copy()
        if not mask.any():
            messagebox.showinfo("Group", "Select at least one electrode first.")
            return
        default = f"Group{len(self.groups) + 1}"
        name = simpledialog.askstring("Group name", "Enter the group name:", initialvalue=default, parent=self)
        if not name:
            return
        color = colorchooser.askcolor(title="Choose group colour", parent=self)[0] or (66, 0, 255)
        self.groups.append({"name": name, "color": tuple(map(int, color)), "mask": mask})
        self.group_list.insert("end", f"{name} ({int(mask.sum())} electrodes)")
        self.clear_selection()

    def delete_group(self):
        selection = self.group_list.curselection()
        if not selection:
            return
        del self.groups[selection[0]]
        self.group_list.delete(selection[0])
        self._redraw()

    def export_json(self):
        if not self.groups:
            messagebox.showinfo("JSON export", "Create at least one electrode group.")
            return
        source = Path(self.path_var.get())
        path = source.with_suffix(".json")
        if path.exists() and not messagebox.askyesno(
            "Replace JSON?", f"{path.name} already exists.\n\nReplace it?"
        ):
            return
        try:
            export_groups_json(path, self.groups)
            if self.registration_matrix is not None and self.slice_image_path:
                registration_path = source.with_name(f"{source.stem}_image_registration.json")
                export_image_registration(
                    registration_path, self.slice_image_path, self.image_landmarks,
                    self.electrode_landmarks, self.registration_matrix,
                    float(self.overlay_opacity_var.get()) / 100,
                )
        except Exception as exc:
            messagebox.showerror("JSON export", str(exc))
            return
        self.status_var.set(f"Exported {len(self.groups)} groups to {path}")
        if self.on_export:
            self.on_export(str(source), str(path))
        messagebox.showinfo("JSON export", f"Saved automatically:\n{path}")
        self._advance()


class BoundaryRegistrationDialog(tk.Toplevel):
    """Collect four manually selected image vertices for the fixed FP-map boundary."""

    CORNER_LABELS = ("1. lower-left", "2. lower-right", "3. upper-right", "4. upper-left")

    def __init__(self, parent: FieldMapFrame, slice_image: np.ndarray,
                 rate_map: np.ndarray, on_apply):
        super().__init__(parent)
        self.slice_image = slice_image
        self.rate_map = rate_map
        self.on_apply = on_apply
        self.image_points: list[tuple[float, float]] = []
        self.point_markers, self.point_labels = [], []
        self.outline = None
        self.title("Manually fit slice image to FP-rate map")
        self.transient(parent.winfo_toplevel())
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._build()

    def _build(self):
        ttk.Label(self, text="Manual slice-image boundary alignment", style="Title.TLabel").pack(
            anchor="w", padx=12, pady=(12, 2)
        )
        self.help_var = tk.StringVar(
            value="Click the four points around the slice image that correspond to the FP-map boundary, in this order: 1 lower-left, 2 lower-right, 3 upper-right, 4 upper-left."
        )
        ttk.Label(self, textvariable=self.help_var, wraplength=900).pack(anchor="w", padx=12, pady=(0, 8))

        self.figure = Figure(figsize=(10.6, 5.2))
        self.image_axis = self.figure.add_axes((0.06, 0.13, 0.40, 0.76))
        self.map_axis = self.figure.add_axes((0.56, 0.13, 0.34, 0.76))
        self.color_axis = self.figure.add_axes((0.92, 0.13, 0.025, 0.76))
        height, width = self.slice_image.shape[:2]
        self.image_axis.imshow(self.slice_image, origin="lower", extent=(0, width - 1, 0, height - 1))
        self.image_axis.set(title="Slice image: select its FP-map limits", xlabel="Image column (pixels)", ylabel="Image row (pixels)")
        self.image_axis.set_xlim(0, width - 1)
        self.image_axis.set_ylim(0, height - 1)
        heatmap = self.map_axis.imshow(self.rate_map, origin="lower")
        self.map_axis.set(title="Fixed FP-map boundary reference", xlabel="Electrode column (0–63)", ylabel="Electrode row (0–63)")
        self.map_axis.set_xlim(-.5, 63.5)
        self.map_axis.set_ylim(-.5, 63.5)
        self.map_axis.add_patch(Rectangle((-.5, -.5), 64, 64, fill=False, edgecolor="white", linewidth=1.4, linestyle="--"))
        for index, (x, y) in enumerate(fp_map_boundary_vertices(), start=1):
            self.map_axis.plot(x, y, marker="o", markersize=6, color="#E5A11A")
            ha = "left" if x < 1 else "right"
            va = "bottom" if y < 1 else "top"
            self.map_axis.text(x, y, f" {index}", color="white", fontsize=9, ha=ha, va=va, fontweight="bold")
        self.figure.colorbar(heatmap, cax=self.color_axis, label="FP rate (events/min)")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
        self.canvas.mpl_connect("button_press_event", self._click)

        actions = ttk.Frame(self, padding=12)
        actions.pack(fill="x")
        ttk.Button(actions, text="Undo last vertex", command=self.undo_last).pack(side="left")
        ttk.Button(actions, text="Reset vertices", command=self.reset_vertices).pack(side="left", padx=5)
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(actions, text="Apply boundary fit", command=self.apply).pack(side="right", padx=(0, 5))

    def _click(self, event):
        if event.inaxes != self.image_axis or event.xdata is None or event.ydata is None:
            return
        if len(self.image_points) >= 4:
            self.help_var.set("All four vertices are selected. Use Apply boundary fit, Undo last vertex, or Reset vertices.")
            return
        point = (float(event.xdata), float(event.ydata))
        self.image_points.append(point)
        index = len(self.image_points)
        self.point_markers.append(self.image_axis.plot(*point, marker="o", markersize=7, markerfacecolor="none",
                                                       markeredgewidth=1.8, markeredgecolor="#E5A11A")[0])
        self.point_labels.append(self.image_axis.text(*point, f" {index}", color="white", fontsize=9,
                                                      ha="left", va="bottom", fontweight="bold"))
        self._draw_outline()
        if index < 4:
            self.help_var.set(f"Vertex {index} selected ({self.CORNER_LABELS[index - 1]}). Next: click {self.CORNER_LABELS[index]}.")
        else:
            self.help_var.set("All four vertices selected. Review the yellow outline, then choose Apply boundary fit.")
        self.canvas.draw_idle()

    def _draw_outline(self):
        if self.outline is not None:
            self.outline.remove()
        if len(self.image_points) > 1:
            ordered = self.image_points + ([self.image_points[0]] if len(self.image_points) == 4 else [])
            self.outline = self.image_axis.plot([x for x, _ in ordered], [y for _, y in ordered], color="#E5A11A", linewidth=1.1)[0]

    def undo_last(self):
        if not self.image_points:
            return
        self.image_points.pop()
        self.point_markers.pop().remove()
        self.point_labels.pop().remove()
        self._draw_outline()
        next_index = len(self.image_points)
        self.help_var.set(f"Click {self.CORNER_LABELS[next_index]} in the slice image.")
        self.canvas.draw_idle()

    def reset_vertices(self):
        for artist in self.point_markers + self.point_labels:
            artist.remove()
        if self.outline is not None:
            self.outline.remove()
        self.image_points, self.point_markers, self.point_labels, self.outline = [], [], [], None
        self.help_var.set("Click the four slice-image boundary vertices in order: 1 lower-left, 2 lower-right, 3 upper-right, 4 upper-left.")
        self.canvas.draw_idle()

    def apply(self):
        targets = fp_map_boundary_vertices()
        try:
            matrix = solve_homography(self.image_points, targets)
        except ValueError as exc:
            messagebox.showerror("Boundary fit", str(exc), parent=self)
            return
        self.on_apply(self.image_points.copy(), targets, matrix)
        self.destroy()
