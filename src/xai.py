from pathlib import Path

import matplotlib.pyplot as plt
import shap

from src.evaluation import plot_message_figure


def generate_shap_summary(pipeline, X_sample, output_path: Path, max_rows: int = 500) -> None:
    """Generate a compact SHAP global explanation for the final model."""
    X_small = X_sample.sample(min(len(X_sample), max_rows), random_state=42)
    preprocess = pipeline.named_steps["preprocess"]
    selector = pipeline.named_steps["select"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocess.transform(X_small)
    X_selected = selector.transform(X_transformed)

    feature_names = preprocess.get_feature_names_out()
    if hasattr(selector, "get_support"):
        feature_names = feature_names[selector.get_support()]

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_selected)
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, X_selected)
        shap_values = explainer.shap_values(X_selected)
    else:
        plot_message_figure(
            output_path,
            "Synthese SHAP globale",
            "SHAP n'est pas disponible pour ce modele. Consultez l'importance par permutation.",
        )
        return

    plt.figure()
    shap.summary_plot(shap_values, X_selected, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
