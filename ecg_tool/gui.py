from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .csv_io import read_ecg_csv
from .extractor import ExtractionSettings, extract_pdf, safe_base_name, write_four_csvs
from .source_preview import preview_source
from .interaction import zoom_limits
from .review import save_review
from .plotting import calculate_full_limits, format_ecg_axes, save_ecg_figure


class ECGApplication(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Samsung ECG PDF to CSV")
        self.geometry("1200x880")
        self.minsize(1000, 720)
        self.option_add("*Font", ("Segoe UI", 10))
        self._events: queue.Queue = queue.Queue()
        self._current_frame = None
        self._full_limits = None
        self._plot_active = False
        self._drag_state = None
        self._build_style()
        self._build_ui()
        self.after(100, self._poll_events)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        available = style.theme_names()
        if "vista" in available:
            style.theme_use("vista")
        elif "clam" in available:
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Accent.TButton", padding=(14, 8))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Samsung ECG Digitizer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Convert a three-strip Samsung Health Monitor ECG PDF into four CSV files, then view and export ECG plots. The zero line is set automatically.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        converter = ttk.Frame(notebook, padding=18)
        viewer = ttk.Frame(notebook, padding=12)
        notebook.add(converter, text="PDF to 4 CSV Files")
        notebook.add(viewer, text="CSV to ECG Image")
        self._build_converter(converter)
        self._build_viewer(viewer)

    def _build_converter(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self.pdf_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.rate_var = tk.StringVar(value="500")
        self.duration_var = tk.StringVar(value="10")
        self.gain_var = tk.StringVar(value="10")
        self.dpi_var = tk.StringVar(value="300")

        self._file_row(parent, 0, "ECG PDF", self.pdf_var, self._choose_pdf, "Browse...")
        self._file_row(parent, 1, "Output Folder", self.output_var, self._choose_output, "Browse...")
        ttk.Label(parent, text="Base File Name").grid(row=2, column=0, sticky="w", pady=8)
        ttk.Entry(parent, textvariable=self.name_var).grid(row=2, column=1, sticky="ew", padx=10, pady=8)
        ttk.Label(parent, text="Leave blank to use the PDF file name", style="Hint.TLabel").grid(row=2, column=2, sticky="w")

        settings = ttk.LabelFrame(parent, text="Digitization Settings", padding=12)
        settings.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        for column in range(8):
            settings.columnconfigure(column, weight=1 if column % 2 else 0)
        entries = [
            ("Sample Rate (Hz)", self.rate_var),
            ("Segment Duration (s)", self.duration_var),
            ("Reference Gain (mm/mV)", self.gain_var),
            ("Render DPI", self.dpi_var),
        ]
        for index, (label, variable) in enumerate(entries):
            ttk.Label(settings, text=label).grid(row=0, column=index * 2, sticky="e", padx=(5, 4))
            ttk.Entry(settings, textvariable=variable, width=9).grid(row=0, column=index * 2 + 1, sticky="w", padx=(0, 12))

        ttk.Label(settings, text="Default 0 mV: second internal major horizontal grid line from bottom; border excluded.").grid(row=1, column=0, columnspan=7, sticky="w", pady=(10, 0))
        ttk.Button(settings, text="Preview Source", command=self._preview_source).grid(row=1, column=7, pady=(10, 0))

        ttk.Label(
            parent,
            text="The calibration pulse is fixed at 1 mV. Always creates four CSVs; missing signal is blank/NaN.",
            style="Hint.TLabel",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 10))
        self.convert_button = ttk.Button(parent, text="Convert PDF", style="Accent.TButton", command=self._start_conversion)
        self.convert_button.grid(row=5, column=0, columnspan=3, pady=8)

        status_box = ttk.LabelFrame(parent, text="Processing Status", padding=10)
        status_box.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        parent.rowconfigure(6, weight=1)
        status_box.rowconfigure(1, weight=1)
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, textvariable=self.status_var).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.log_text = tk.Text(status_box, height=10, wrap="word", state="disabled", relief="flat", background="#fafafa")
        self.log_text.grid(row=1, column=0, sticky="nsew")

    def _build_viewer(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)
        self.csv_var = tk.StringVar()
        self.plot_status_var = tk.StringVar(value="Load a CSV file. Hover over the plot and scroll to zoom.")
        self.image_dir_var = tk.StringVar()
        self.image_name_var = tk.StringVar(value="ecg_plot")
        self.image_format_var = tk.StringVar(value="PNG")
        self.export_dpi_var = tk.StringVar(value="300")
        self.image_width_var = tk.StringVar(value="12")
        self.image_height_var = tk.StringVar(value="6")
        self.save_range_var = tk.StringVar(value="Current View")
        self.background_var = tk.StringVar(value="ECG Grid")
        self.plot_title_var = tk.StringVar()
        self.line_color_var = tk.StringVar(value="#b23a2b")
        self.line_width_var = tk.StringVar(value="0.85")

        ttk.Label(parent, text="CSV File").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        ttk.Entry(parent, textvariable=self.csv_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(parent, text="Browse...", command=self._choose_csv).grid(row=0, column=2, padx=8, pady=(0, 8))
        ttk.Button(parent, text="Plot ECG", style="Accent.TButton", command=self._plot_csv).grid(row=0, column=3, pady=(0, 8))
        ttk.Button(parent, text="Reset View", command=self._reset_view).grid(row=0, column=4, padx=(8, 0), pady=(0, 8))

        plot_frame = ttk.Frame(parent, relief="solid", borderwidth=1)
        plot_frame.grid(row=1, column=0, columnspan=5, sticky="nsew")
        self.figure = Figure(figsize=(9, 5.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        format_ecg_axes(self.axes, "Select an ECG CSV file", "ECG Grid")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

        ttk.Label(parent, textvariable=self.plot_status_var, style="Hint.TLabel").grid(row=2, column=0, columnspan=5, sticky="w", pady=(5, 8))
        self._build_export_settings(parent)

    def _build_export_settings(self, parent: ttk.Frame) -> None:
        export = ttk.LabelFrame(parent, text="Image Export Settings", padding=10)
        export.grid(row=3, column=0, columnspan=5, sticky="ew")
        for column in (1, 4, 7):
            export.columnconfigure(column, weight=1)

        ttk.Label(export, text="Save Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(export, textvariable=self.image_dir_var).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(export, text="Browse...", command=self._choose_image_folder).grid(row=0, column=4, sticky="w")
        ttk.Label(export, text="File Name").grid(row=0, column=5, sticky="e", padx=(12, 4))
        ttk.Entry(export, textvariable=self.image_name_var, width=24).grid(row=0, column=6, columnspan=2, sticky="ew")

        ttk.Label(export, text="Format").grid(row=1, column=0, sticky="e", pady=7)
        ttk.Combobox(export, textvariable=self.image_format_var, values=("PNG", "JPG", "SVG", "PDF"), state="readonly", width=7).grid(row=1, column=1, sticky="w", padx=(6, 12), pady=7)
        ttk.Label(export, text="DPI").grid(row=1, column=2, sticky="e")
        ttk.Combobox(export, textvariable=self.export_dpi_var, values=("100", "150", "300", "600"), width=7).grid(row=1, column=3, sticky="w", padx=(6, 12))
        ttk.Label(export, text="Size (inches)").grid(row=1, column=4, sticky="e")
        size_frame = ttk.Frame(export)
        size_frame.grid(row=1, column=5, sticky="w", padx=(6, 12))
        ttk.Entry(size_frame, textvariable=self.image_width_var, width=6).pack(side="left")
        ttk.Label(size_frame, text=" x ").pack(side="left")
        ttk.Entry(size_frame, textvariable=self.image_height_var, width=6).pack(side="left")
        ttk.Label(export, text="Save Range").grid(row=1, column=6, sticky="e")
        ttk.Combobox(export, textvariable=self.save_range_var, values=("Current View", "Full Signal"), state="readonly", width=13).grid(row=1, column=7, sticky="w", padx=(6, 0))

        ttk.Label(export, text="Background").grid(row=2, column=0, sticky="e")
        background = ttk.Combobox(export, textvariable=self.background_var, values=("ECG Grid", "White"), state="readonly", width=10)
        background.grid(row=2, column=1, sticky="w", padx=(6, 12))
        background.bind("<<ComboboxSelected>>", self._refresh_plot_style)
        ttk.Label(export, text="Title").grid(row=2, column=2, sticky="e")
        ttk.Entry(export, textvariable=self.plot_title_var).grid(row=2, column=3, columnspan=2, sticky="ew", padx=(6, 12))
        ttk.Label(export, text="Line Color").grid(row=2, column=5, sticky="e")
        ttk.Button(export, text="Choose...", command=self._choose_line_color).grid(row=2, column=6, sticky="w", padx=(6, 12))
        ttk.Label(export, text="Line Width").grid(row=2, column=7, sticky="e")
        ttk.Entry(export, textvariable=self.line_width_var, width=7).grid(row=2, column=8, sticky="w", padx=(6, 0))

        ttk.Button(export, text="Save ECG Image", style="Accent.TButton", command=self._save_ecg_image).grid(row=3, column=0, columnspan=9, pady=(10, 0))

    def _file_row(self, parent, row, label, variable, callback, button_text) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=8)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=10, pady=8)
        ttk.Button(parent, text=button_text, command=callback).grid(row=row, column=2, sticky="ew", pady=8)

    def _choose_pdf(self) -> None:
        path = filedialog.askopenfilename(title="Select Samsung ECG PDF", filetypes=[("PDF Files", "*.pdf")])
        if path:
            self.pdf_var.set(path)
            if not self.name_var.get().strip():
                self.name_var.set(Path(path).stem)
            if not self.output_var.get():
                self.output_var.set(str(Path(path).parent))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Select CSV Output Folder")
        if path:
            self.output_var.set(path)

    def _choose_csv(self) -> None:
        path = filedialog.askopenfilename(title="Select ECG CSV", filetypes=[("CSV Files", "*.csv")])
        if path:
            csv_path = Path(path)
            self.csv_var.set(path)
            self.image_dir_var.set(str(csv_path.parent))
            self.image_name_var.set(f"{csv_path.stem}_plot")
            self.plot_title_var.set(csv_path.stem)

    def _choose_image_folder(self) -> None:
        path = filedialog.askdirectory(title="Select Image Output Folder")
        if path:
            self.image_dir_var.set(path)

    def _choose_line_color(self) -> None:
        _, color = colorchooser.askcolor(color=self.line_color_var.get(), title="Select ECG Line Color", parent=self)
        if color:
            self.line_color_var.set(color)
            if self.axes.lines:
                self.axes.lines[0].set_color(color)
                self.canvas.draw_idle()

    def _preview_source(self) -> None:
        try:
            path = Path(self.pdf_var.get())
            if not path.is_file():
                raise ValueError("Select a valid ECG PDF first.")
            settings = ExtractionSettings(render_dpi=int(self.dpi_var.get()), gain_mm_per_mV=float(self.gain_var.get()))
            settings.validate()
            preview_source(self, path, settings)
        except Exception as exc:
            messagebox.showerror("Source Preview", str(exc), parent=self)

    def _start_conversion(self) -> None:
        try:
            pdf_path = Path(self.pdf_var.get())
            output_dir = Path(self.output_var.get())
            if not pdf_path.is_file():
                raise ValueError("Select a valid ECG PDF first.")
            if not self.output_var.get().strip():
                raise ValueError("Select an output folder.")
            base_name = safe_base_name(self.name_var.get(), pdf_path.stem)
            settings = ExtractionSettings(sample_rate_hz=int(self.rate_var.get()), segment_duration_s=float(self.duration_var.get()), gain_mm_per_mV=float(self.gain_var.get()), render_dpi=int(self.dpi_var.get()))
            settings.validate()
            planned = [output_dir / f"{base_name}_part{i}.csv" for i in range(1, 4)]
            planned.append(output_dir / f"{base_name}_total.csv")
            planned.extend([output_dir / f"{base_name}_overlay.png", output_dir / f"{base_name}_settings.json"])
            existing = [path.name for path in planned if path.exists()]
            if existing and not messagebox.askyesno("Confirm Overwrite", "The following files already exist. Overwrite them?\n\n" + "\n".join(existing), parent=self):
                return
        except (ValueError, OSError) as exc:
            messagebox.showerror("Invalid Input", str(exc), parent=self)
            return

        self.convert_button.configure(state="disabled")
        self.status_var.set("Converting PDF. Please wait...")
        self._set_log("")
        threading.Thread(target=self._convert_worker, args=(pdf_path, output_dir, base_name, settings), daemon=True).start()

    def _convert_worker(self, pdf_path, output_dir, base_name, settings) -> None:
        try:
            result = extract_pdf(pdf_path, settings)
            paths = write_four_csvs(result, output_dir, base_name)
            try:
                paths.extend(save_review(pdf_path, settings, result, output_dir, base_name))
            except Exception as exc:
                result.diagnostics.append(f"CSV conversion succeeded; review output failed: {exc}")
            self._events.put(("success", result.diagnostics, paths))
        except Exception as exc:
            self._events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == "success":
                    _, diagnostics, paths = event
                    self.convert_button.configure(state="normal")
                    self.status_var.set("Conversion completed: 4 CSV files created")
                    self._set_log("\n".join([*diagnostics, "", "Output files:", *(str(path) for path in paths)]))
                    messagebox.showinfo("Conversion Completed", "Four CSV files were created successfully.", parent=self)
                else:
                    self.convert_button.configure(state="normal")
                    self.status_var.set("Conversion failed")
                    self._set_log(event[1])
                    messagebox.showerror("Conversion Failed", event[1], parent=self)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _set_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", text)
        self.log_text.configure(state="disabled")

    def _plot_csv(self) -> None:
        try:
            frame = read_ecg_csv(self.csv_var.get())
            line_width = self._validated_line_width()
        except ValueError as exc:
            messagebox.showerror("Unable to Plot", str(exc), parent=self)
            return
        self._current_frame = frame
        self.axes.clear()
        self.axes.plot(frame["time_s"], frame["voltage_mV"], color=self.line_color_var.get(), linewidth=line_width)
        title = self.plot_title_var.get().strip() or Path(self.csv_var.get()).name
        format_ecg_axes(self.axes, title, self.background_var.get())
        self._full_limits = calculate_full_limits(frame)
        self.axes.set_xlim(self._full_limits[0])
        self.axes.set_ylim(self._full_limits[1])
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._plot_active = False
        missing = int(frame["voltage_mV"].isna().sum())
        self.plot_status_var.set(f"Plot loaded ({missing} missing samples shown as gaps). Hover over the plot and scroll to zoom. Drag to pan; double-click to reset.")

    def _on_plot_press(self, event) -> None:
        if event.inaxes is not self.axes or self._current_frame is None:
            return
        self._plot_active = True
        self.canvas.get_tk_widget().focus_set()
        self.plot_status_var.set("Plot activated. Scroll to zoom around the pointer; drag to pan; double-click to reset.")
        if event.dblclick:
            self._reset_view()
            return
        if event.button == MouseButton.LEFT:
            self._drag_state = (event.x, event.y, self.axes.get_xlim(), self.axes.get_ylim())

    def _on_plot_release(self, _event) -> None:
        self._drag_state = None

    def _on_plot_motion(self, event) -> None:
        if self._drag_state is None or event.inaxes is not self.axes:
            return
        start_x, start_y, x_limits, y_limits = self._drag_state
        bbox = self.axes.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        data_dx = (event.x - start_x) / bbox.width * (x_limits[1] - x_limits[0])
        data_dy = (event.y - start_y) / bbox.height * (y_limits[1] - y_limits[0])
        self.axes.set_xlim(x_limits[0] - data_dx, x_limits[1] - data_dx)
        self.axes.set_ylim(y_limits[0] - data_dy, y_limits[1] - data_dy)
        self.canvas.draw_idle()

    def _on_plot_scroll(self, event) -> None:
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        self.axes.set_xlim(zoom_limits(self.axes.get_xlim(), event.xdata, event.step))
        self.axes.set_ylim(zoom_limits(self.axes.get_ylim(), event.ydata, event.step))
        self.canvas.draw_idle()

    def _reset_view(self) -> None:
        if self._full_limits is None:
            return
        self.axes.set_xlim(self._full_limits[0])
        self.axes.set_ylim(self._full_limits[1])
        self.canvas.draw_idle()
        self.plot_status_var.set("Full signal view restored.")

    def _refresh_plot_style(self, _event=None) -> None:
        if self._current_frame is None:
            return
        title = self.plot_title_var.get().strip() or Path(self.csv_var.get()).name
        x_limits = self.axes.get_xlim()
        y_limits = self.axes.get_ylim()
        format_ecg_axes(self.axes, title, self.background_var.get())
        self.axes.set_xlim(x_limits)
        self.axes.set_ylim(y_limits)
        self.canvas.draw_idle()

    def _validated_line_width(self) -> float:
        try:
            value = float(self.line_width_var.get())
        except ValueError as exc:
            raise ValueError("Line width must be a number.") from exc
        if not 0.1 <= value <= 10:
            raise ValueError("Line width must be between 0.1 and 10.")
        return value

    def _save_ecg_image(self) -> None:
        if self._current_frame is None:
            messagebox.showerror("Unable to Save", "Plot an ECG CSV first.", parent=self)
            return
        try:
            output_text = self.image_dir_var.get().strip()
            if not output_text:
                raise ValueError("Select an image output folder.")
            output_dir = Path(output_text)
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = safe_base_name(self.image_name_var.get(), "ecg_plot")
            image_format = self.image_format_var.get().lower()
            dpi = int(self.export_dpi_var.get())
            width = float(self.image_width_var.get())
            height = float(self.image_height_var.get())
            line_width = self._validated_line_width()
            if not 50 <= dpi <= 1200:
                raise ValueError("Image DPI must be between 50 and 1200.")
            if not 2 <= width <= 40 or not 2 <= height <= 30:
                raise ValueError("Image width and height are outside the allowed range.")
            output_path = output_dir / f"{base_name}.{image_format}"
            if output_path.exists() and not messagebox.askyesno("Confirm Overwrite", f"{output_path.name} already exists. Overwrite it?", parent=self):
                return
        except (ValueError, OSError) as exc:
            messagebox.showerror("Invalid Export Settings", str(exc), parent=self)
            return

        if self.save_range_var.get() == "Current View":
            x_limits = self.axes.get_xlim()
            y_limits = self.axes.get_ylim()
        else:
            x_limits = self._full_limits[0]
            y_limits = self._full_limits[1]
        try:
            save_ecg_figure(
                self._current_frame,
                output_path,
                image_format=image_format,
                dpi=dpi,
                width=width,
                height=height,
                title=self.plot_title_var.get().strip(),
                line_color=self.line_color_var.get(),
                line_width=line_width,
                background=self.background_var.get(),
                x_limits=x_limits,
                y_limits=y_limits,
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Unable to Save", str(exc), parent=self)
            return
        messagebox.showinfo("Image Saved", f"ECG image saved to:\n{output_path}", parent=self)

def main() -> None:
    app = ECGApplication()
    app.mainloop()
