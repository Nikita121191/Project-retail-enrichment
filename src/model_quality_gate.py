import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required report not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")

    return data


def require_number(report: dict, key: str, report_name: str) -> float:
    if key not in report:
        raise KeyError(f"{report_name}: missing metric '{key}'")

    value = report[key]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{report_name}: metric '{key}' must be numeric, got {type(value).__name__}"
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{report_name}: metric '{key}' must be finite, got {value}"
        )

    return value


def add_min_check(
    checks: list[dict[str, Any]],
    *,
    model: str,
    metric: str,
    value: float,
    threshold: float,
) -> None:
    passed = value >= threshold
    checks.append({
        "model": model,
        "metric": metric,
        "operator": ">=",
        "threshold": float(threshold),
        "value": float(value),
        "passed": bool(passed),
    })


def add_max_check(
    checks: list[dict[str, Any]],
    *,
    model: str,
    metric: str,
    value: float,
    threshold: float,
) -> None:
    passed = value <= threshold
    checks.append({
        "model": model,
        "metric": metric,
        "operator": "<=",
        "threshold": float(threshold),
        "value": float(value),
        "passed": bool(passed),
    })


def evaluate_quality(
    domain_report: dict,
    founded_report: dict,
    price_report: dict,
    policy: dict,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    # -------------------------
    # Domain
    # -------------------------
    domain_cfg = policy["domain"]

    add_min_check(
        checks,
        model="domain",
        metric="cv_best_macro_f1",
        value=require_number(domain_report, "cv_best_macro_f1", "domain_report"),
        threshold=domain_cfg["min_cv_macro_f1"],
    )
    add_min_check(
        checks,
        model="domain",
        metric="holdout_macro_f1",
        value=require_number(domain_report, "holdout_macro_f1", "domain_report"),
        threshold=domain_cfg["min_holdout_macro_f1"],
    )
    add_min_check(
        checks,
        model="domain",
        metric="cv_uplift_over_baseline",
        value=require_number(
            domain_report,
            "cv_uplift_over_baseline",
            "domain_report",
        ),
        threshold=domain_cfg["min_cv_uplift_over_baseline"],
    )
    add_max_check(
        checks,
        model="domain",
        metric="cv_std_macro_f1",
        value=require_number(domain_report, "cv_std_macro_f1", "domain_report"),
        threshold=domain_cfg["max_cv_std_macro_f1"],
    )
    add_min_check(
        checks,
        model="domain",
        metric="generalization_gap_macro_f1",
        value=require_number(
            domain_report,
            "generalization_gap_macro_f1",
            "domain_report",
        ),
        threshold=domain_cfg["min_generalization_gap_macro_f1"],
    )

    # -------------------------
    # Founded
    # -------------------------
    founded_cfg = policy["founded"]

    add_min_check(
        checks,
        model="founded",
        metric="cv_mae_improvement_over_baseline_pct",
        value=require_number(
            founded_report,
            "cv_mae_improvement_over_baseline_pct",
            "founded_report",
        ),
        threshold=founded_cfg["min_cv_mae_improvement_over_baseline_pct"],
    )
    add_max_check(
        checks,
        model="founded",
        metric="holdout_mae",
        value=require_number(founded_report, "holdout_mae", "founded_report"),
        threshold=founded_cfg["max_holdout_mae"],
    )
    add_min_check(
        checks,
        model="founded",
        metric="holdout_r2",
        value=require_number(founded_report, "holdout_r2", "founded_report"),
        threshold=founded_cfg["min_holdout_r2"],
    )
    add_max_check(
        checks,
        model="founded",
        metric="holdout_p90_absolute_error",
        value=require_number(
            founded_report,
            "holdout_p90_absolute_error",
            "founded_report",
        ),
        threshold=founded_cfg["max_holdout_p90_absolute_error"],
    )
    add_max_check(
        checks,
        model="founded",
        metric="generalization_gap_mae",
        value=require_number(
            founded_report,
            "generalization_gap_mae",
            "founded_report",
        ),
        threshold=founded_cfg["max_generalization_gap_mae"],
    )

    # -------------------------
    # Price category
    # -------------------------
    price_cfg = policy["price_category"]

    add_min_check(
        checks,
        model="price_category",
        metric="cv_best_macro_f1",
        value=require_number(price_report, "cv_best_macro_f1", "price_report"),
        threshold=price_cfg["min_cv_macro_f1"],
    )
    add_min_check(
        checks,
        model="price_category",
        metric="cv_uplift_over_baseline",
        value=require_number(
            price_report,
            "cv_uplift_over_baseline",
            "price_report",
        ),
        threshold=price_cfg["min_cv_uplift_over_baseline"],
    )
    add_min_check(
        checks,
        model="price_category",
        metric="holdout_serving_macro_f1",
        value=require_number(
            price_report,
            "holdout_serving_macro_f1",
            "price_report",
        ),
        threshold=price_cfg["min_holdout_serving_macro_f1"],
    )
    add_min_check(
        checks,
        model="price_category",
        metric="holdout_serving_micro_f1",
        value=require_number(
            price_report,
            "holdout_serving_micro_f1",
            "price_report",
        ),
        threshold=price_cfg["min_holdout_serving_micro_f1"],
    )
    add_min_check(
        checks,
        model="price_category",
        metric="holdout_serving_samples_f1",
        value=require_number(
            price_report,
            "holdout_serving_samples_f1",
            "price_report",
        ),
        threshold=price_cfg["min_holdout_serving_samples_f1"],
    )
    add_min_check(
        checks,
        model="price_category",
        metric="generalization_gap_macro_f1",
        value=require_number(
            price_report,
            "generalization_gap_macro_f1",
            "price_report",
        ),
        threshold=price_cfg["min_generalization_gap_macro_f1"],
    )

    direct_macro = require_number(
        price_report,
        "holdout_direct_macro_f1",
        "price_report",
    )
    serving_macro = require_number(
        price_report,
        "holdout_serving_macro_f1",
        "price_report",
    )

    serving_drop = direct_macro - serving_macro

    add_max_check(
        checks,
        model="price_category",
        metric="serving_macro_drop_vs_direct",
        value=serving_drop,
        threshold=price_cfg["max_serving_macro_drop_vs_direct"],
    )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply model-quality policy before model promotion."
    )
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    training_dir = args.training_dir

    domain_report = load_json(
        training_dir / "domain" / "domain_report.json"
    )
    founded_report = load_json(
        training_dir / "founded" / "founded_report.json"
    )
    price_report = load_json(
        training_dir / "price_category" / "price_category_report.json"
    )
    policy = load_json(args.config_path)

    required_policy_sections = {
        "policy_version",
        "domain",
        "founded",
        "price_category",
    }
    missing_policy_sections = required_policy_sections - set(policy)

    if missing_policy_sections:
        raise KeyError(
            "Quality policy is missing sections: "
            f"{sorted(missing_policy_sections)}"
        )

    checks = evaluate_quality(
        domain_report=domain_report,
        founded_report=founded_report,
        price_report=price_report,
        policy=policy,
    )

    failures = [check for check in checks if not check["passed"]]

    gate_report = {
        "policy_version": policy["policy_version"],
        "status": "PASSED" if not failures else "FAILED",
        "training_dir": str(training_dir),
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failures),
        "checks_failed": len(failures),
        "checks": checks,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(
        json.dumps(
            gate_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n==============================")
    print("MODEL QUALITY GATE")
    print("==============================")
    print(f"Policy version : {policy['policy_version']}")

    for check in checks:
        marker = "PASS" if check["passed"] else "FAIL"
        print(
            f"[{marker}] "
            f"{check['model']}.{check['metric']} "
            f"= {check['value']:.6f} "
            f"{check['operator']} {check['threshold']:.6f}"
        )

    print("------------------------------")
    print(
        f"Result         : {gate_report['status']} "
        f"({gate_report['checks_passed']}/{gate_report['checks_total']} checks passed)"
    )
    print(f"Report saved to: {args.out_path}")

    if failures:
        failed_names = [
            f"{check['model']}.{check['metric']}"
            for check in failures
        ]
        raise RuntimeError(
            "Model quality gate failed: "
            + ", ".join(failed_names)
        )


if __name__ == "__main__":
    main()
