"""Write reproducible settings and a source-image overlay without filling gaps."""
from dataclasses import asdict
from pathlib import Path
import json
import re
import numpy as np
from matplotlib.figure import Figure
from .extractor import _load_first_page, _orient_page


def save_review(pdf_path, settings, result, output_dir, base_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image, _, _ = _load_first_page(Path(pdf_path), settings.render_dpi)
    image, _, bands, _ = _orient_page(image)
    zero_lines = []
    for note in result.diagnostics:
        match = re.search(r"zero line page y=([\d.]+)", note)
        if match:
            zero_lines.append(float(match.group(1)))
    scale_note = next(note for note in result.diagnostics if "using " in note and "px/mV" in note)
    scale = float(re.search(r"using ([\d.]+) px/mV", scale_note).group(1))
    figure = Figure(figsize=(15, 7), dpi=150)
    for i, (frame, (y0, y1, x0, x1), baseline) in enumerate(zip(result.segments, bands, zero_lines)):
        ax = figure.add_subplot(3, 1, i+1)
        ax.imshow(image[y0:y1, x0:x1], extent=(x0-.5, x1-.5, y1-.5, y0-.5), aspect="auto")
        xx = x0 + frame.time_s.to_numpy() / settings.segment_duration_s * (x1-x0)
        yy = baseline-frame.voltage_mV.to_numpy()*scale
        ax.plot(xx, yy, color="#007fa8", linewidth=.6, alpha=.8, label="CSV reconstruction")
        ax.axhline(baseline, color="red", linewidth=.9, label="default 0 mV")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"Segment {i+1}: source orange / CSV blue / default zero red", fontsize=10)
        ax.set_xticks([])
        if i==0:
            ax.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    overlay = output_dir / f"{base_name}_overlay.png"
    figure.savefig(overlay)
    metadata = output_dir / f"{base_name}_settings.json"
    metadata.write_text(json.dumps({
        "program_version": "1.7.0", "source_name": Path(pdf_path).name,
        "settings": asdict(settings), "bands_y0_y1_x0_x1": bands,
        "zero_lines_page_y_px": zero_lines,
        "diagnostics": result.diagnostics,
        "data_kind": "image-digitized approximation; not original device samples"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return [overlay, metadata]
