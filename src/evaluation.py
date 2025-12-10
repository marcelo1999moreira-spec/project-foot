import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE and R² for regression.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def plot_real_vs_pred(y_true, y_pred, title="Real vs Predicted"):
    """
    Plot real vs predicted values on a scatterplot.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("Real value (M€)")
    plt.ylabel("Predicted value (M€)")
    plt.title(title)
    plt.show()
