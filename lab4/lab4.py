import numpy as np
import matplotlib.pyplot as plt
import warnings
import os


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(filename):
    filepath = f"data/{filename}"
    try:
        return np.loadtxt(filepath)
    except FileNotFoundError:
        print(f"File {filepath} not found")
        return None


def calculate_dw(residuals):
    return np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)


def build_trend_model(y, k, order=1):
    if order == 1:
        X = np.column_stack([np.ones(len(k)), k])
        model_name = "Trend order 1"
    else:
        X = np.column_stack([np.ones(len(k)), k, k**2])
        model_name = "Trend order 2"

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ coeffs
    residuals = y - y_pred

    n = len(y)
    k_vars = X.shape[1]

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    ss_regression = ss_total - ss_residual

    r_squared = ss_regression / ss_total
    rmse = np.sqrt(ss_residual / (n - k_vars))
    mse = np.mean(residuals**2)
    f_stat = (ss_regression / (k_vars - 1)) / (ss_residual / (n - k_vars))

    var_covar = np.linalg.inv(X.T @ X) * rmse**2
    se_coeffs = np.sqrt(np.diag(var_covar))
    t_stats = coeffs / se_coeffs

    dw = calculate_dw(residuals)

    return {
        "coefficients": coeffs,
        "predictions": y_pred,
        "residuals": residuals,
        "model_name": model_name,
        "order": order,
        "r_squared": r_squared,
        "rmse": rmse,
        "mse": mse,
        "f_stat": f_stat,
        "se_coeffs": se_coeffs,
        "t_stats": t_stats,
        "ss_total": ss_total,
        "ss_residual": ss_residual,
        "ss_regression": ss_regression,
        "dw": dw,
    }


def print_model_summary(results, dataset_name):
    print(f"\n{dataset_name} - {results['model_name']}")
    print("-" * 70)

    if results["order"] == 1:
        print(
            f"Equation: y(k) = {results['coefficients'][0]:.8f} + {results['coefficients'][1]:.8f}*k"
        )
        print(
            f"Coefficients: c(1) = {results['coefficients'][0]:.8f}, c(2) = {results['coefficients'][1]:.8f}"
        )
    else:
        print(
            f"Equation: y(k) = {results['coefficients'][0]:.8f} + {results['coefficients'][1]:.8f}*k + {results['coefficients'][2]:.8f}*k²"
        )
        print(
            f"Coefficients: c(1) = {results['coefficients'][0]:.8f}, c(2) = {results['coefficients'][1]:.8f}, c(3) = {results['coefficients'][2]:.8f}"
        )

    print(f"\nStatistics:")
    print(f"  R² = {results['r_squared']:.8f}")
    print(f"  Sum Sq Resid = {results['ss_residual']:.4f}")
    print(f"  DW = {results['dw']:.8f}")
    print(f"  RMSE = {results['rmse']:.8f}")
    print(f"  F-stat = {results['f_stat']:.6f}")


def plot_trend(y, y_pred, title, filename):
    plt.figure(figsize=(12, 6))
    t = np.arange(1, len(y) + 1)

    plt.plot(t, y, "b-", label="Real data", linewidth=2)
    plt.plot(t, y_pred, "r--", label="Trend", linewidth=2)

    plt.xlabel("Time (k)")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()


def plot_residuals(residuals, title, filename):
    plt.figure(figsize=(12, 6))
    t = np.arange(1, len(residuals) + 1)

    plt.plot(t, residuals, "g-", linewidth=1.5)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=1)

    plt.xlabel("Time (k)")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()


def process_trend_models(y_train, k_train, dataset_name):
    """Build and compare trend models"""
    res1 = build_trend_model(y_train, k_train, order=1)
    res2 = build_trend_model(y_train, k_train, order=2)

    print_model_summary(res1, dataset_name)
    print_model_summary(res2, dataset_name)

    dir1 = f"results/{dataset_name}/order1"
    dir2 = f"results/{dataset_name}/order2"
    create_dir(dir1)
    create_dir(dir2)

    plot_trend(
        y_train,
        res1["predictions"],
        f"{dataset_name}: Time series and Trend (Order 1)",
        f"{dir1}/trend.png",
    )
    plot_residuals(
        res1["residuals"],
        f"{dataset_name}: Residuals (Order 1)",
        f"{dir1}/residuals.png",
    )

    plot_trend(
        y_train,
        res2["predictions"],
        f"{dataset_name}: Time series and Trend (Order 2)",
        f"{dir2}/trend.png",
    )
    plot_residuals(
        res2["residuals"],
        f"{dataset_name}: Residuals (Order 2)",
        f"{dir2}/residuals.png",
    )

    best = res1 if res1["r_squared"] > res2["r_squared"] else res2
    return best


def forecast_and_plot(best, y_train, y_test, dataset_name):
    """Generate forecast and save plot"""
    order = best["order"]
    k_test = np.arange(len(y_train) + 1, len(y_train) + len(y_test) + 1)

    if order == 1:
        X_test = np.column_stack([np.ones(len(k_test)), k_test])
    else:
        X_test = np.column_stack([np.ones(len(k_test)), k_test, k_test**2])

    y_pred = X_test @ best["coefficients"]
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([best["predictions"], y_pred])

    forecast_dir = f"results/{dataset_name}/forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: Forecast (Best model - {best['model_name']})",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    warnings.filterwarnings("ignore")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10

    if not os.path.exists("results"):
        os.makedirs("results")

    # Main execution
    print("\n" + "=" * 70)
    print("LAB 4 - VARIANT 9 - TREND MODELING")
    print("=" * 70)

    datasets = {"IMPGE": "IMPGE.txt", "CURRNS": "CURRNS.txt"}

    for dataset_name, filename in datasets.items():
        y = load_data(filename)
        if y is None:
            continue

        print(f"\n\nProcessing: {dataset_name}")
        print(f"Sample size: {len(y)}")

        train_size = len(y) - 8
        y_train = y[:train_size]
        y_test = y[train_size:]
        k_train = np.arange(1, len(y_train) + 1)

        best = process_trend_models(y_train, k_train, dataset_name)
        print(f"\nBest model: {best['model_name']} (R² = {best['r_squared']:.8f})")

        test_rmse = forecast_and_plot(best, y_train, y_test, dataset_name)
        print(f"Test RMSE: {test_rmse:.8f}")
        print(f"Plots saved to 'results' folder")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70 + "\n")
