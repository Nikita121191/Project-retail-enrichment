import argparse
import subprocess
import sys
from pathlib import Path


AUDIT_SCRIPTS = [
    "audit_schema_and_missing.py",
    "audit_name.py",
    "audit_country.py",
    "audit_domain.py",
    "audit_price_category.py",
    "audit_founded.py",
    "audit_presence.py",
    "audit_description.py",
    "audit_total_rented_area.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/kaggle/input/datasets/nikitasadovoy/russian-retail/russian_retail.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/kaggle/working/preprocessing_audit",
    )
    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(".").resolve()

    for script_name in AUDIT_SCRIPTS:
        script_path = base_dir / script_name
        print(f"\n=== Running {script_name} ===")
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--csv_path", args.csv_path,
                "--out_dir", str(out_dir),
            ],
            check=True
        )

    print("\nAll audits completed.")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
