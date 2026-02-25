import shap
import matplotlib.pyplot as plt
import os

def explain_model(model, X):
    os.makedirs("outputs", exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM returns list for binary classification
    if isinstance(shap_values, list):
     shap_values = shap_values[1]

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
    plt.close()

    print("SHAP plots saved")