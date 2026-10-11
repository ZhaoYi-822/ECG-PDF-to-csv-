import math
from pathlib import Path
import tempfile
import unittest

import fitz
import numpy as np

from ecg_tool.extractor import (
    ExtractionSettings,
    _load_first_page,
    _detect_calibration_pulse_px,
    _orient_page,
    _render_first_page,
    extract_pdf,
)


class SyntheticExtractionTests(unittest.TestCase):
    def test_one_millivolt_pulse_uses_stroke_center_height(self):
        image = np.full((400, 500, 3), 255, dtype=np.uint8)
        image[100:103, 50:103] = 40
        image[200:203, 50:103] = 40
        image[100:203, 50:53] = 40
        image[100:203, 100:103] = 40
        band = (80, 280, 50, 450)
        self.assertEqual(_detect_calibration_pulse_px(image, band, 100.0), 100.0)
        self.assertIsNone(_detect_calibration_pulse_px(np.full_like(image, 255), band, 100.0))
        self.assertIsNone(_detect_calibration_pulse_px(image, band, 50.0))

    def test_missing_calibration_pulse_stops_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "no_pulse.pdf"
            self._make_pdf(pdf_path, include_calibration=False)
            with self.assertRaisesRegex(ValueError, "No reliable 1 mV calibration pulse"):
                extract_pdf(pdf_path, ExtractionSettings(render_dpi=200))

    def test_three_strips_become_continuous_total(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "synthetic.pdf"
            self._make_pdf(pdf_path)
            result = extract_pdf(
                pdf_path,
                ExtractionSettings(
                    baseline_mode="auto",
                    sample_rate_hz=100,
                    segment_duration_s=10,
                    render_dpi=200,
                ),
            )
            self.assertEqual(len(result.segments), 3)
            self.assertTrue(all(len(part) == 1000 for part in result.segments))
            self.assertEqual(len(result.total), 3000)
            self.assertAlmostEqual(result.total["time_s"].iloc[-1], 29.99)
            self.assertTrue(all(list(part.columns) == ["time_s", "voltage_mV"] for part in result.segments))
            self.assertGreater(result.segments[0]["voltage_mV"].isna().sum(), 0)
            self.assertEqual(result.segments[1]["voltage_mV"].isna().sum(), 0)

    def test_rotated_portrait_report_is_turned_clockwise(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "synthetic.pdf"
            self._make_pdf(pdf_path)
            landscape = _render_first_page(pdf_path, 200)
            portrait_with_sideways_content = np.rot90(landscape, 1).copy()
            upright, _, bands, note = _orient_page(portrait_with_sideways_content)
            self.assertGreater(upright.shape[1], upright.shape[0])
            self.assertEqual(len(bands), 3)
            self.assertIn("clockwise", note)

    def test_embedded_tiles_are_assembled_without_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "tiles.pdf"
            document = fitz.open()
            page = document.new_page(width=100, height=100)
            colors = [(220, 30, 30), (30, 220, 30), (30, 30, 220), (220, 220, 30)]
            for index, (x, y) in enumerate(((0, 0), (50, 0), (0, 50), (50, 50))):
                samples = np.full((50, 50, 3), colors[index], dtype=np.uint8)
                pixmap = fitz.Pixmap(fitz.csRGB, 50, 50, samples.tobytes(), False)
                page.insert_image(fitz.Rect(x, y, x + 50, y + 50), pixmap=pixmap)
            document.save(pdf_path)
            document.close()

            image, dpi, note = _load_first_page(pdf_path, 300)
            self.assertEqual(image.shape, (100, 100, 3))
            self.assertAlmostEqual(dpi, 72.0)
            self.assertIn("assembled native", note)
            self.assertTrue(np.array_equal(image[25, 25], colors[0]))
            self.assertTrue(np.array_equal(image[75, 75], colors[3]))

    @staticmethod
    def _make_pdf(path: Path, include_calibration: bool = True) -> None:
        document = fitz.open()
        page = document.new_page(width=842, height=595)
        orange = (0.70, 0.24, 0.12)
        pale_grid = (0.88, 0.88, 0.88)
        for index, baseline in enumerate((180, 300, 420)):
            for x in np.arange(55, 790, 8):
                page.draw_line((float(x), baseline - 45), (float(x), baseline + 45), color=pale_grid, width=0.5)
            for y in np.arange(baseline - 40, baseline + 41, 8):
                page.draw_line((55, float(y)), (790, float(y)), color=pale_grid, width=0.5)
            points = []
            signal_start = 90 if index == 0 else 55
            for x in np.linspace(signal_start, 790, 1800):
                phase = ((x - 55) % 73.5) / 73.5
                signal = 1.3 * math.sin(phase * math.tau)
                signal += -8 * math.exp(-((phase - 0.38) / 0.025) ** 2)
                signal += 30 * math.exp(-((phase - 0.42) / 0.018) ** 2)
                signal += -11 * math.exp(-((phase - 0.46) / 0.022) ** 2)
                signal += 6 * math.exp(-((phase - 0.68) / 0.10) ** 2)
                points.append((float(x), float(baseline - signal)))
            page.draw_polyline(points, color=orange, width=1.25)
            if index == 0 and include_calibration:
                page.draw_polyline(
                    [(55, baseline), (55, baseline - 30), (75, baseline - 30), (75, baseline), (55, baseline)],
                    color=(0.25, 0.25, 0.25),
                    width=1.0,
                )
        document.save(path)
        document.close()


if __name__ == "__main__":
    unittest.main()
