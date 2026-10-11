from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ecg_tool.csv_io import read_ecg_csv
from ecg_tool.extractor import ExtractionResult, safe_base_name, write_four_csvs


def _result() -> ExtractionResult:
    segments = []
    for _ in range(3):
        segments.append(pd.DataFrame({"time_s": [0.0, 0.5], "voltage_mV": [0.1, -0.2]}))
    total = pd.concat(
        [part.assign(time_s=part.time_s + index) for index, part in enumerate(segments)],
        ignore_index=True,
    )
    return ExtractionResult(segments, total, [])


class CoreTests(unittest.TestCase):
    def test_write_four_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_four_csvs(_result(), Path(directory), "ECG_001")
            self.assertEqual(
                [path.name for path in paths],
                [
                    "ECG_001_part1.csv",
                    "ECG_001_part2.csv",
                    "ECG_001_part3.csv",
                    "ECG_001_total.csv",
                ],
            )
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(
                list(pd.read_csv(paths[-1]).columns), ["time_s", "voltage_mV"]
            )

    def test_safe_windows_filename(self):
        self.assertEqual(safe_base_name('ECG:001/abc', "fallback"), "ECG_001_abc")
        self.assertEqual(safe_base_name("", "record"), "record")

    def test_read_ecg_csv_validates_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame({"time": [0, 1], "value": [0, 1]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "time_s"):
                read_ecg_csv(path)

    def test_read_ecg_csv_requires_increasing_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_time.csv"
            pd.DataFrame({"time_s": [0, 0], "voltage_mV": [0, 1]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                read_ecg_csv(path)

    def test_read_ecg_csv_allows_missing_voltage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_voltage.csv"
            path.write_text(
                "time_s,voltage_mV\n0.000,\n0.002,0.12\n0.004,0.10\n",
                encoding="utf-8",
            )
            frame = read_ecg_csv(path)
            self.assertTrue(pd.isna(frame.loc[0, "voltage_mV"]))
            self.assertEqual(frame["voltage_mV"].notna().sum(), 2)


if __name__ == "__main__":
    unittest.main()
