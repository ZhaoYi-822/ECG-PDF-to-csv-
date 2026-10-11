# Samsung ECG PDF to CSV v1.7

A Python desktop application based on the uploaded v1.4 example. All application labels, dialogs and documentation are in English.

## Peak amplitude correction in v1.7

The previous global path could shortcut steep waveform tips and reduce R-peak amplitude. This release detects visible upper and lower turning points in the colored raster trace, estimates stroke-center positions from the line width, and constrains the path to pass through those positions. Horizontal post-smoothing was removed. No gain multiplier or prescribed ECG peak amplitude is applied.

The first R peak in the supplied PDF changed from 0.243586 mV in v1.6 to 0.382353 mV in v1.7 at 0.834 s, consistent with a source-grid reading of approximately 0.38-0.40 mV. The comparison image and example CSVs are not included in the public repository because ECG recordings may contain private health data.

## Other retained behavior

- The zero-voltage reference is automatic and fixed: the **second internal major horizontal grid line counted upward from the bottom** of each ECG strip. The bottom border is excluded; minor grid lines are excluded. This is two major squares above the bottom border.
- No manual zero-line selection or confirmation is required. Conversion does not estimate zero from the waveform.
- Source Preview displays the detected reference as a red horizontal line. The preview is read-only.
- Hover over the source image or CSV plot and use the mouse wheel to zoom around the pointer. No activation click is required. Drag with the left button to pan; double-click to reset.
- Four CSV files, a source overlay PNG and a settings JSON are produced.

## Installation and launch

Use Python 3.10 or later with tkinter. The first installation needs internet access.

Windows: extract the archive, install Python, then double-click `start_windows.bat`. It creates a local virtual environment, installs dependencies and launches the application. If `py` is unavailable, install Python with the Python launcher or follow the commands below with `python`.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Some Linux installations require the system tkinter package. The repository contains source code, not a prebuilt Windows EXE. Run `build_windows.bat` on Windows if you want to build an EXE.

## Convert a report

1. Select the Samsung Health Monitor ECG PDF, output folder and base file name.
2. Keep the defaults for the provided report: 500 Hz, 10 seconds per strip, reference gain 10 mm/mV, fallback rendering 300 DPI.
3. Optionally click **Preview Source** to inspect the fixed red zero lines. Scroll to zoom; drag to pan. Close the preview when finished.
4. Click **Convert PDF**. The program identifies the default grid reference in each strip automatically. If the required grid or calibration pulse cannot be verified, conversion stops rather than guessing a zero line.
5. Review the output overlay: orange is the original waveform, blue is the CSV reconstruction and red is the default zero reference.
6. Use the **CSV to ECG Image** tab to load, zoom, pan and export a CSV plot.

For the provided report, each small grid square represents 0.04 seconds horizontally and 0.1 mV vertically. Each major square represents 0.2 seconds and 0.5 mV. The zero reference is therefore approximately 1 mV above the bottom border. Positive voltage is above the reference; negative voltage is below it.

The black 1 mV calibration pulse determines the voltage scale. Its measured stroke-center height is cross-checked against the grid and physical PDF dimensions. The reference line determines only the voltage offset. Time is mapped over the full 10-second chart width, including the leading calibration-only area.

## Output files

- `{name}_part1.csv`, `{name}_part2.csv`, `{name}_part3.csv`: local time 0–9.998 s, 5000 rows per strip at 500 Hz.
- `{name}_total.csv`: continuous time 0–29.998 s, 15000 rows.
- Columns: `time_s,voltage_mV`. Missing voltage is blank/NaN; it is never replaced with zero.
- `{name}_overlay.png`: source-image overlay for visual review.
- `{name}_settings.json`: settings, detected grid references, image bounds and diagnostics.

The three default zero lines in the uploaded PDF are at upright image y coordinates 826.5, 1255.5 and 1684.5 px. These are detected from the grid, not hardcoded coordinates. Example outputs are intentionally excluded from the public repository; generate your own outputs locally.

The GUI asks before replacing existing output files. The command-line tool overwrites matching output names. If review-image export fails after successful CSV creation, the GUI log reports the review failure separately.

## Command line

Use the default fixed grid reference:

```bash
python convert_cli.py input.pdf --output output --name ECG
```

Reuse settings from the same PDF and layout:

```bash
python convert_cli.py input.pdf --output output --settings output/ECG_settings.json
```

Advanced compatibility options `--auto-baseline` and `--baselines F1 F2 F3` remain available only in the CLI. They explicitly override the default; the GUI has no manual baseline controls. Avoid reusing v1.5 settings if you want the new default, since saved manual references override it.

## Scope and accuracy

The program handles three orange waveform strips on the first page of Samsung Health Monitor ECG reports. Other layouts may not be supported. Long gaps and edge regions remain missing; only internal gaps of at most 20 ms are interpolated.

Values are approximations reconstructed from a plotted image, not original watch samples. A 500 Hz CSV contains interpolation. Image resolution, line thickness, overlapping strokes and resampling still limit recoverable peak values. Amplitude tests cover selected synthetic raster cases, not every possible recording. The fixed grid reference preserves baseline drift rather than automatically removing it.

## Validation

```bash
python -m unittest discover -s tests -v
```

See `VALIDATION.md` for the results and limitations of testing in this environment.

## Privacy and repository contents

The public repository excludes `example_output/`, patient ECG PDFs, generated CSV files, overlays, settings JSON files, and local Python environments. Use your own ECG reports locally; do not commit identifiable health data.

