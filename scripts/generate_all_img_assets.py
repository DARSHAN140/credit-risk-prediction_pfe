from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


IMAGE_GROUPS = {
    "eda": [
        "generate_realistic_eda_images.py",
    ],
    "preprocessing": [
        "generate_feature_engineering_images.py",
    ],
    "feature": [
        "generate_feature_engineering_images.py",
    ],
    "training": [
        "generate_model_comparison_images.py",
        "generate_best_model_evaluation_images.py",
    ],
    "evaluation": [
        "generate_model_comparison_images.py",
        "generate_best_model_evaluation_images.py",
    ],
    "workflow": [
        "generate_workflow_images.py",
    ],
    "architecture": [
        "generate_architecture_diagram.py",
    ],
    "governance": [
        "generate_governance_images.py",
    ],
    "docker": [
        "generate_docker_configuration_image.py",
    ],
    "docker-scripts": [
        "generate_docker_scripts_image.py",
    ],
    "auc-comparison": [
        "generate_auc_comparison_image.py",
    ],
    "real-security-governance": [
        "generate_real_security_governance_images.py",
    ],
    "roc-curves-comparison": [
        "generate_roc_curves_comparison_image.py",
    ],
    "metrics": [
        "generate_metrics_curves.py",
        "generate_real_auc_ks_threshold_curves.py",
    ],
    "xgboost-reporting": [
        "generate_xgboost_reporting_curves.py",
    ],
    "fastapi-interface": [
        "generate_fastapi_interface_capture.py",
    ],
}

HEAVY_GROUPS = {
    "per-model": [
        "generate_per_model_evaluation_images.py",
    ],
    "correlation-full": [
        "generate_full_correlation_matrix.py",
        "generate_model_correlation_matrix.py",
    ],
}


def script_path(script_name: str) -> Path:
    return BASE_DIR / "scripts" / script_name


def run_script(script_name: str) -> None:
    path = script_path(script_name)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    print(f"\n=== Running {script_name} ===")
    subprocess.run(
        [PYTHON, str(path)],
        cwd=BASE_DIR,
        check=True,
    )


def resolve_scripts(groups: list[str], include_heavy: bool) -> list[str]:
    requested = groups or [
        "eda",
        "preprocessing",
        "training",
        "workflow",
        "architecture",
        "governance",
        "docker",
        "docker-scripts",
        "auc-comparison",
        "real-security-governance",
        "roc-curves-comparison",
        "metrics",
        "fastapi-interface",
    ]
    available = dict(IMAGE_GROUPS)
    if include_heavy:
        available.update(HEAVY_GROUPS)

    scripts: list[str] = []
    for group in requested:
        if group == "all":
            for names in IMAGE_GROUPS.values():
                scripts.extend(names)
            if include_heavy:
                for names in HEAVY_GROUPS.values():
                    scripts.extend(names)
            continue

        if group not in available:
            valid = ", ".join(sorted([*IMAGE_GROUPS, *HEAVY_GROUPS, "all"]))
            raise ValueError(f"Unknown group '{group}'. Valid groups: {valid}")
        scripts.extend(available[group])

    unique_scripts = []
    for script in scripts:
        if script not in unique_scripts:
            unique_scripts.append(script)
    return unique_scripts


def list_outputs() -> None:
    img_dir = BASE_DIR / "img"
    print("\nGenerated images in img/:")
    for path in sorted(img_dir.glob("*.png")):
        print(f" - {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate project report images into the img directory.",
    )
    parser.add_argument(
        "--group",
        nargs="*",
        default=[],
        help=(
            "Image groups to generate: eda, preprocessing, feature, training, "
            "evaluation, workflow, architecture, governance, docker, docker-scripts, "
            "auc-comparison, real-security-governance, roc-curves-comparison, "
            "metrics, xgboost-reporting, fastapi-interface, all. "
            "Default generates the main report images."
        ),
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help=(
            "Also allow heavy groups such as per-model and correlation-full. "
            "These can retrain models and take longer."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List PNG files currently available in img after generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = resolve_scripts(args.group, args.include_heavy)

    print("Output directory:", BASE_DIR / "img")
    print("Groups:", ", ".join(args.group or ["default"]))

    for script in scripts:
        run_script(script)

    if args.list:
        list_outputs()

    print("\nDone. All selected images were generated in img/.")


if __name__ == "__main__":
    main()
