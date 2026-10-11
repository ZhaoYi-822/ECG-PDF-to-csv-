from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ecg_tool.plotting import calculate_full_limits, save_ecg_figure


class PlottingTests(unittest.TestCase):
    def test_plot_with_nan_can_be_saved(self):
        frame = pd.DataFrame(
            {
                "time_s": np.arange(0, 1, 0.01),
                "voltage_mV": np.sin(np.arange(0, 1, 0.01) * 10),
            }
        )
        frame.loc[:9, "voltage_mV"] = np.nan
        x_limits, y_limits = calculate_full_limits(frame)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ecg.png"
            save_ecg_figure(
                frame,
                output,
                image_format="png",
                dpi=100,
                width=6,
                height=3,
                title="ECG Plot",
                line_color="#b23a2b",
                line_width=0.85,
                background="ECG Grid",
                x_limits=x_limits,
                y_limits=y_limits,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
