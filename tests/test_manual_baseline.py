import tempfile
import unittest
from pathlib import Path
import numpy as np
from ecg_tool.extractor import ExtractionSettings, extract_pdf, _load_first_page, _orient_page
import test_extraction_synthetic as synthetic


class ManualBaselineTests(unittest.TestCase):
    def test_invalid_manual_reference_rejected(self):
        for values in ((0.5, 0.5), (0.5, float('nan'), 0.5), (0.5, 1.01, 0.5)):
            with self.assertRaises(ValueError):
                ExtractionSettings(baseline_fractions=values).validate()

    def test_manual_shift_preserves_waveform_and_missing_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory)/'signal.pdf'
            synthetic.SyntheticExtractionTests._make_pdf(pdf)
            im, dpi, _ = _load_first_page(pdf, 200)
            _, _, bands, _ = _orient_page(im)
            a = extract_pdf(pdf, ExtractionSettings(sample_rate_hz=100, render_dpi=200, baseline_fractions=(.5,.5,.5)))
            b = extract_pdf(pdf, ExtractionSettings(sample_rate_hz=100, render_dpi=200, baseline_fractions=(.6,.6,.6)))
            # Synthetic PDF calibration pulse is 30 PDF points, i.e. 30/72*dpi px/mV.
            pulse = 30/72*dpi
            for i, (fa, fb) in enumerate(zip(a.segments, b.segments)):
                np.testing.assert_array_equal(fa.voltage_mV.isna(), fb.voltage_mV.isna())
                difference = (fb.voltage_mV-fa.voltage_mV).dropna().to_numpy()
                self.assertLess(np.ptp(difference), 1e-10)
                expected = .1*(bands[i][1]-bands[i][0]-1)/pulse
                self.assertAlmostEqual(float(difference[0]), expected, delta=.015)
            np.testing.assert_array_equal(a.total.time_s, b.total.time_s)
