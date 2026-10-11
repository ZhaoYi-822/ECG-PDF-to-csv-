"""Noninteractive conversion with a fixed default grid zero line."""
import argparse
from pathlib import Path
from ecg_tool.extractor import ExtractionSettings, extract_pdf, safe_base_name, write_four_csvs
from ecg_tool.review import save_review


def main():
    parser = argparse.ArgumentParser(description="Samsung ECG PDF to CSV v1.7")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--name", default="")
    parser.add_argument("--settings", type=Path, help="Reuse a generated settings JSON for the SAME PDF/layout")
    parser.add_argument("--baselines", type=float, nargs=3, metavar=("F1", "F2", "F3"), help="Zero-line fractions from strip top, 0..1")
    parser.add_argument("--auto-baseline", action="store_true", help="Explicitly use automatic estimated baselines")
    args = parser.parse_args()
    if sum((args.settings is not None, args.baselines is not None, args.auto_baseline)) > 1:
        parser.error("Choose at most one of --settings, --baselines, --auto-baseline")
    if args.settings:
        import json
        data = json.loads(args.settings.read_text(encoding="utf-8"))
        values = data["settings"]
        if values.get("baseline_fractions") is not None:
            values["baseline_fractions"] = tuple(values["baseline_fractions"])
        settings = ExtractionSettings(**values)
    else:
        settings = ExtractionSettings(baseline_mode="auto" if args.auto_baseline else "second_grid", baseline_fractions=tuple(args.baselines) if args.baselines else None)
    result = extract_pdf(args.pdf, settings)
    name = safe_base_name(args.name, args.pdf.stem)
    paths = write_four_csvs(result, args.output, name)
    paths.extend(save_review(args.pdf, settings, result, args.output, name))
    print("\n".join(result.diagnostics))
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
