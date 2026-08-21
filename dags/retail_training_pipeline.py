import json
import re
import subprocess
import sys
from pathlib import Path

from airflow.sdk import dag, get_current_context, task


# ============================================================
# Project paths inside Docker
# ============================================================

PROJECT_DIR = Path("/opt/airflow/project")

SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

INPUT_CSV = DATA_DIR / "russian_retail.csv"

RUN_ALL_AUDITS_SCRIPT = SRC_DIR / "run_all_audits.py"
QA_CHECKS_SCRIPT = SRC_DIR / "qa_checks.py"
TRAIN_DOMAIN_SCRIPT = SRC_DIR / "train_domain.py"
TRAIN_FOUNDED_SCRIPT = SRC_DIR / "train_founded.py"
TRAIN_PRICE_SCRIPT = SRC_DIR / "train_price_category.py"

VALIDATE_ARTIFACTS_SCRIPT = (
    SRC_DIR / "validate_artifacts.py"
)
MODEL_CONTRACT_SCRIPT = (
    SRC_DIR / "model_contract.py"
)
MODEL_QUALITY_GATE_SCRIPT = (
    SRC_DIR / "model_quality_gate.py"
)

AIRFLOW_CONFIG_DIR = Path("/opt/airflow/config")

MODEL_QUALITY_CONFIG = (
    AIRFLOW_CONFIG_DIR / "model_quality_thresholds.json"
)
SMOKE_INFERENCE_SCRIPT = (
    SRC_DIR / "smoke_inference.py"
)
PUBLISH_MODELS_SCRIPT = (
    SRC_DIR / "publish_models.py"
)

CURRENT_ARTIFACTS_DIR = (
    ARTIFACTS_DIR / "current"
)

SMOKE_SAMPLE_SIZE = 20
TEST_SIZE = 0.2
PRICE_VALIDATION_SIZE = 0.2
TRAIN_N_JOBS = 2


# ============================================================
# Helpers
# ============================================================

def make_safe_run_id(run_id: str) -> str:
    """
    Converts Airflow run_id into a filesystem-friendly directory name.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="retail_training_pipeline",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["retail", "training"],
)
def retail_training_pipeline():

    # --------------------------------------------------------
    # 1. Check input
    # --------------------------------------------------------

    @task
    def check_input() -> str:
        required_paths = [
            INPUT_CSV,
            RUN_ALL_AUDITS_SCRIPT,
            QA_CHECKS_SCRIPT,
            MODEL_CONTRACT_SCRIPT,
            TRAIN_DOMAIN_SCRIPT,
            TRAIN_FOUNDED_SCRIPT,
            TRAIN_PRICE_SCRIPT,
            VALIDATE_ARTIFACTS_SCRIPT,
            MODEL_QUALITY_GATE_SCRIPT,
            MODEL_QUALITY_CONFIG,
            SMOKE_INFERENCE_SCRIPT,
            PUBLISH_MODELS_SCRIPT,
        ]

        missing = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Required project files are missing: {missing}"
            )

        print("Input validation passed.")

        for path in required_paths:
            print(f"OK: {path}")

        return str(INPUT_CSV)

    # --------------------------------------------------------
    # 2A. Run detailed audits
    # --------------------------------------------------------

    @task
    def run_all_audits(csv_path: str) -> str:
        context = get_current_context()

        run_id = make_safe_run_id(
            str(context["run_id"])
        )

        out_dir = (
            ARTIFACTS_DIR
            / "pipeline_runs"
            / run_id
            / "audits"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            sys.executable,
            str(RUN_ALL_AUDITS_SCRIPT),
            "--csv_path",
            csv_path,
            "--out_dir",
            str(out_dir),
        ]

        print("Running audits:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )

        print(f"Audit artifacts saved to: {out_dir}")

        return str(out_dir)

    # --------------------------------------------------------
    # 2B. Run training QA
    # --------------------------------------------------------

    @task
    def run_qa_checks(csv_path: str) -> str:
        context = get_current_context()

        run_id = make_safe_run_id(
            str(context["run_id"])
        )

        out_dir = (
            ARTIFACTS_DIR
            / "pipeline_runs"
            / run_id
            / "qa"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            sys.executable,
            str(QA_CHECKS_SCRIPT),
            "--csv_path",
            csv_path,
            "--out_dir",
            str(out_dir),
        ]

        print("Running QA checks:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )

        print(f"QA artifacts saved to: {out_dir}")

        return str(out_dir)

    # --------------------------------------------------------
    # Dependencies
    # --------------------------------------------------------
    # --------------------------------------------------------
    # 3. Quality gate
    # --------------------------------------------------------

    @task
    def quality_gate(audit_dir: str, qa_dir: str) -> str:
        audit_path = Path(audit_dir)
        qa_path = Path(qa_dir)

        qa_summary_path = qa_path / "qa_summary.json"

        # ----------------------------------------------------
        # Technical validation
        # ----------------------------------------------------

        if not audit_path.is_dir():
            raise FileNotFoundError(
                f"Audit directory does not exist: {audit_path}"
            )

        if not qa_summary_path.is_file():
            raise FileNotFoundError(
                f"QA summary not found: {qa_summary_path}"
            )

        with qa_summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        failures = []
        warnings = []

        # ----------------------------------------------------
        # Dataset / schema checks
        # ----------------------------------------------------

        required_check = report.get(
            "required_columns_check",
            {},
        )

        if required_check.get("status") != "OK":
            failures.append(
                "Required columns check failed: "
                f"{required_check.get('missing_required_columns', [])}"
            )

        schema_report = report.get(
            "schema_report",
            {},
        )

        missing_canonical = schema_report.get(
            "missing_canonical_columns",
            [],
        )

        if missing_canonical:
            failures.append(
                "Missing canonical columns: "
                f"{missing_canonical}"
            )

        raw_shape = report.get(
            "dataset_shape_raw",
            [],
        )

        prepared_shape = report.get(
            "dataset_shape_prepared",
            [],
        )

        if len(raw_shape) < 2 or raw_shape[0] <= 0:
            failures.append(
                f"Raw dataset is empty or invalid: {raw_shape}"
            )

        if len(prepared_shape) < 2 or prepared_shape[0] <= 0:
            failures.append(
                f"Prepared dataset is empty or invalid: {prepared_shape}"
            )

        if (
            len(raw_shape) >= 2
            and len(prepared_shape) >= 2
            and raw_shape[0] != prepared_shape[0]
        ):
            failures.append(
                "Row count changed during preparation: "
                f"{raw_shape[0]} -> {prepared_shape[0]}"
            )

        # ----------------------------------------------------
        # Training readiness
        # ----------------------------------------------------

        readiness = report.get(
            "training_readiness",
            {},
        )

        domain = readiness.get(
            "domain_task",
            {},
        )

        founded = readiness.get(
            "founded_task",
            {},
        )

        price = readiness.get(
            "price_category_task",
            {},
        )

        if domain.get("rows_after_drop_rare_classes", 0) <= 0:
            failures.append(
                "No usable rows for domain training."
            )

        if domain.get("n_classes_ge_5", 0) < 2:
            failures.append(
                "Domain training requires at least "
                "two sufficiently represented classes."
            )

        if founded.get("rows_in_valid_range_1850_2025", 0) <= 1:
            failures.append(
                "Not enough valid rows for founded training."
            )

        if price.get("rows_after_drop_unknown_only", 0,) <= 0:
            failures.append(
                "No usable rows for "
                "price_category training."
            )
        if price.get("unexpected_labels", []):
            failures.append(
                "Unexpected price_category labels: "
                f"{price['unexpected_labels']}"
            )
        if price.get("min_known_label_support", 0,) < 5:
            failures.append(
                "At least one known price label "
                "has fewer than 5 rows."
            )

        # ----------------------------------------------------
        # Non-blocking anomalies
        # ----------------------------------------------------

        anomalies = report.get(
            "anomaly_report",
            {},
        )

        if anomalies.get("founded_lt_1850", 0) > 0:
            warnings.append(
                "Founded values below 1850: "
                f"{anomalies['founded_lt_1850']}"
            )

        if anomalies.get("founded_missing", 0) > 0:
            warnings.append(
                "Missing founded values: "
                f"{anomalies['founded_missing']}"
            )

        if anomalies.get("description_empty", 0) > 0:
            warnings.append(
                "Empty descriptions: "
                f"{anomalies['description_empty']}"
            )

        # ----------------------------------------------------
        # Report result
        # ----------------------------------------------------

        print("\n=== QUALITY GATE REPORT ===")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"WARNING: {warning}")

        if failures:
            print("\nBlocking failures:")
            for failure in failures:
                print(f"FAILED: {failure}")

            raise RuntimeError(
                "Quality gate failed. "
                "Training is not allowed."
            )

        print("\nQUALITY GATE PASSED.")
        print("Training is allowed.")

        return str(qa_path)

        # --------------------------------------------------------
    # 4A. Train domain model
    # --------------------------------------------------------

    @task
    def train_domain(
        csv_path: str,
        gate_result: str,
    ) -> str:
        context = get_current_context()

        run_id = make_safe_run_id(
            str(context["run_id"])
        )

        out_dir = (
            ARTIFACTS_DIR
            / "pipeline_runs"
            / run_id
            / "training"
            / "domain"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"Quality gate passed: {gate_result}")

        command = [
            sys.executable,
            str(TRAIN_DOMAIN_SCRIPT),
            "--csv_path",
            csv_path,
            "--out_dir",
            str(out_dir),
            "--test_size",
            str(TEST_SIZE),
            "--n_jobs",
            str(TRAIN_N_JOBS),
        ]

        print("Running domain training:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )

        print(
            f"Domain artifacts saved to: {out_dir}"
        )

        return str(out_dir)

        # --------------------------------------------------------
    # 4B. Train founded model
    # --------------------------------------------------------

    @task
    def train_founded(
        csv_path: str,
        gate_result: str,
    ) -> str:
        context = get_current_context()

        run_id = make_safe_run_id(
            str(context["run_id"])
        )

        out_dir = (
            ARTIFACTS_DIR
            / "pipeline_runs"
            / run_id
            / "training"
            / "founded"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"Quality gate passed: {gate_result}")

        command = [
            sys.executable,
            str(TRAIN_FOUNDED_SCRIPT),
            "--csv_path",
            csv_path,
            "--out_dir",
            str(out_dir),
            "--test_size",
            str(TEST_SIZE),
            "--n_jobs",
            str(TRAIN_N_JOBS),
        ]

        print("Running founded training:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )

        print(
            f"Founded artifacts saved to: {out_dir}"
        )

        return str(out_dir)

        # --------------------------------------------------------
    # 4C. Train price category model
    # --------------------------------------------------------

    @task
    def train_price_category(
        csv_path: str,
        gate_result: str,
    ) -> str:
        context = get_current_context()

        run_id = make_safe_run_id(
            str(context["run_id"])
        )

        out_dir = (
            ARTIFACTS_DIR
            / "pipeline_runs"
            / run_id
            / "training"
            / "price_category"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"Quality gate passed: {gate_result}")

        command = [
            sys.executable,
            str(TRAIN_PRICE_SCRIPT),
            "--csv_path",
            csv_path,
            "--out_dir",
            str(out_dir),
            "--test_size",
            str(TEST_SIZE),
            "--validation_size",
            str(PRICE_VALIDATION_SIZE),
            "--n_jobs",
            str(TRAIN_N_JOBS),
        ]

        print("Running price category training:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )

        print(
            "Price category artifacts saved to: "
            f"{out_dir}"
        )

        return str(out_dir)

        # --------------------------------------------------------
    # 5. Validate training artifacts
    # --------------------------------------------------------

    @task
    def validate_artifacts(
        domain_dir: str,
        founded_dir: str,
        price_dir: str,
    ) -> str:
        command = [
            sys.executable,
            str(VALIDATE_ARTIFACTS_SCRIPT),

            "--domain_dir",
            domain_dir,

            "--founded_dir",
            founded_dir,

            "--price_dir",
            price_dir,
        ]

        print("Validating training artifacts:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )

        training_dir = Path(domain_dir).parent

        print(
            "Training artifacts validated successfully."
        )

        return str(training_dir)
    @task
    def model_quality_gate(
        training_dir: str,
    ) -> str: 
        training_path = Path(training_dir)
        out_path = (
            training_path.parent
            / "model_quality_gate_report.json"
        )
        command = [
            sys.executable,
            str(MODEL_QUALITY_GATE_SCRIPT),
            "--training_dir",
            training_dir,
            "--config_path",
            str(MODEL_QUALITY_CONFIG),
            "--out_path",
            str(out_path),
        ]

        print("Running model quality gate:")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
        )
        print(
            "Model quality gate passed."
        )
        return str(out_path)
    @task
    def smoke_inference(
        csv_path: str,
        training_dir: str,
        quality_report: str,
    ) -> str:
        context = get_current_context()

        run_id = make_safe_run_id(
            str(context["run_id"])
        )

        out_dir = (
            ARTIFACTS_DIR
            / "pipeline_runs"
            / run_id
            / "smoke_inference"
        )
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        print(
            "Model quality gate passed: "
            f"{quality_report}"
        )

        command = [
            sys.executable,
            str(SMOKE_INFERENCE_SCRIPT),
            "--csv_path",
            csv_path,
            "--artifacts_dir",
            training_dir,
            "--out_dir",
            str(out_dir),
            "--sample_size",
            str(SMOKE_SAMPLE_SIZE),
        ]
        print("Running smoke inference:")
        print(" ".join(command))
        subprocess.run(
            command,
            check=True,
        )
        report_path = (
            out_dir
            / "smoke_inference_report.json"
        )
        print(
            "Smoke inference passed."
        )
        return str(report_path)

    @task
    def publish_models(
        training_dir: str,
        quality_report: str,
        smoke_report: str,
    ) -> str:
        context = get_current_context()
        run_id = str(
            context["run_id"]
        )
        command = [
            sys.executable,
            str(PUBLISH_MODELS_SCRIPT),
            "--training_dir",
            training_dir,
            "--quality_report",
            quality_report,
            "--smoke_report",
            smoke_report,
            "--publish_dir",
            str(CURRENT_ARTIFACTS_DIR),
            "--source_run_id",
            run_id,
        ]
        print("Publishing model bundle:")
        print(" ".join(command))
        subprocess.run(
            command,
            check=True,
        )
        manifest_path = (
            CURRENT_ARTIFACTS_DIR
            / "manifest.json"
        )
        print(
            "Models published successfully."
        )
        return str(manifest_path)
    csv_path = check_input()

    audit_dir = run_all_audits(csv_path)
    qa_dir = run_qa_checks(csv_path)

    gate_result = quality_gate(
        audit_dir,
        qa_dir,
    )
    domain_dir = train_domain(
        csv_path,
        gate_result,
    )
    founded_dir = train_founded(
        csv_path,
        gate_result,
    )
    price_dir = train_price_category(
        csv_path,
        gate_result,
    )
    training_dir = validate_artifacts(
        domain_dir,
        founded_dir,
        price_dir,
    )
    
    quality_report = model_quality_gate(
        training_dir,
    )
    smoke_report = smoke_inference(
        csv_path,
        training_dir,
        quality_report,
    )
    manifest_path = publish_models(
        training_dir,
        quality_report,
        smoke_report,
    )

retail_training_pipeline()