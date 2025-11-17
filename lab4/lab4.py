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


def build_ar_model(y, p=1):
    """AR(p) model"""
    n = len(y)

    X = np.ones((n - p, 1))
    for i in range(1, p + 1):
        X = np.column_stack([X, y[p - i : -i if i > 0 else None]])

    y_ar = y[p:]

    coeffs = np.linalg.lstsq(X, y_ar, rcond=None)[0]
    y_pred = X @ coeffs
    residuals = y_ar - y_pred

    ss_residual = np.sum(residuals**2)
    rmse = np.sqrt(ss_residual / (len(y_ar) - X.shape[1]))
    r_squared = 1 - ss_residual / np.sum((y_ar - np.mean(y_ar)) ** 2)
    dw = calculate_dw(residuals)

    return {
        "coefficients": coeffs,
        "predictions": y_pred,
        "residuals": residuals,
        "order": p,
        "r_squared": r_squared,
        "rmse": rmse,
        "ss_residual": ss_residual,
        "dw": dw,
    }


def build_arma_model(y, p=1, q=1):
    """ARMA(p,q) model - simplified"""
    n = len(y)

    X = np.ones((n - max(p, q), 1))
    for i in range(1, p + 1):
        X = np.column_stack([X, y[max(p, q) - i : -i if i > 0 else None]])

    y_arma = y[max(p, q) :]

    coeffs = np.linalg.lstsq(X, y_arma, rcond=None)[0]
    y_pred = X @ coeffs
    residuals = y_arma - y_pred

    ss_residual = np.sum(residuals**2)
    rmse = np.sqrt(ss_residual / (len(y_arma) - X.shape[1]))
    r_squared = 1 - ss_residual / np.sum((y_arma - np.mean(y_arma)) ** 2)
    dw = calculate_dw(residuals)

    return {
        "coefficients": coeffs,
        "predictions": y_pred,
        "residuals": residuals,
        "order": (p, q),
        "r_squared": r_squared,
        "rmse": rmse,
        "ss_residual": ss_residual,
        "dw": dw,
    }


def build_arima_model(y, order=(1, 1, 1)):
    """ARIMA(p,d,q) model - simplified version"""
    p, d, q = order

    diff_y = y.copy()
    for _ in range(d):
        diff_y = np.diff(diff_y)

    n = len(diff_y)

    X = np.ones((n - max(p, q), 1))
    for i in range(1, p + 1):
        X = np.column_stack([X, diff_y[max(p, q) - i : -i if i > 0 else None]])

    y_ar = diff_y[max(p, q) :]

    if X.shape[1] > 1:
        coeffs = np.linalg.lstsq(X, y_ar, rcond=None)[0]
    else:
        coeffs = np.array([np.mean(y_ar)])

    residuals = y_ar - (X @ coeffs)

    ss_residual = np.sum(residuals**2)
    rmse = np.sqrt(ss_residual / (len(y_ar) - X.shape[1]))
    r_squared = 1 - ss_residual / np.sum((y_ar - np.mean(y_ar)) ** 2)
    dw = calculate_dw(residuals)

    return {
        "coefficients": coeffs,
        "residuals": residuals,
        "order": order,
        "diff_y": diff_y,
        "r_squared": r_squared,
        "rmse": rmse,
        "ss_residual": ss_residual,
        "dw": dw,
    }


def print_model_summary(results, dataset_name, model_type="trend"):
    print(f"\n{dataset_name} - {model_type}")
    print("-" * 70)

    if model_type == "trend":
        if results["order"] == 1:
            print(
                f"y(k) = {results['coefficients'][0]:.8f} + {results['coefficients'][1]:.8f}*k"
            )
        else:
            print(
                f"y(k) = {results['coefficients'][0]:.8f} + {results['coefficients'][1]:.8f}*k + {results['coefficients'][2]:.8f}*k²"
            )
    else:
        print(f"ARIMA{results['order']}")

    print(f"R² = {results['r_squared']:.8f}")
    print(f"Sum Sq Resid = {results['ss_residual']:.4f}")
    print(f"DW = {results['dw']:.8f}")
    print(f"RMSE = {results['rmse']:.8f}")


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


def plot_model_fit(y, y_pred, title, filename, start_idx=0):
    """Generic plot for model fit"""
    plt.figure(figsize=(12, 6))
    t = np.arange(len(y))

    plt.plot(t, y, "b-", label="Real data", linewidth=2)
    plt.plot(t[start_idx:], y_pred, "r--", label="Model fit", linewidth=2)
    plt.xlabel("Time (k)")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()


def process_trend_models(y_train, k_train, dataset_name):
    """Build and compare trend models"""
    res1 = build_trend_model(y_train, k_train, order=1)
    res2 = build_trend_model(y_train, k_train, order=2)

    print_model_summary(res1, dataset_name, "Trend order 1")
    print_model_summary(res2, dataset_name, "Trend order 2")

    dir1 = f"results/{dataset_name}/trend_order1"
    dir2 = f"results/{dataset_name}/trend_order2"
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


def forecast_trend(best, y_train, y_test, dataset_name):
    """Generate trend forecast"""
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

    forecast_dir = f"results/{dataset_name}/trend_forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: Trend Forecast",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


def process_model_with_fit(
    y_train, dataset_name, build_func, func_args, model_dir, model_name
):
    """Generic function to process and plot any model"""
    model = build_func(*func_args)
    print_model_summary(model, dataset_name, model_name)

    create_dir(model_dir)
    plot_residuals(
        model["residuals"],
        f"{dataset_name}: {model_name} Residuals",
        f"{model_dir}/residuals.png",
    )

    plot_model_fit(
        y_train,
        model["predictions"],
        f"{dataset_name}: {model_name} Model Fit",
        f"{model_dir}/fit.png",
        start_idx=func_args[1] if len(func_args) > 1 else 0,
    )

    return model


def process_ar_model(y_train, dataset_name):
    """Build AR(4) model"""
    model_dir = f"results/{dataset_name}/ar_model_p4"
    return process_model_with_fit(
        y_train, dataset_name, build_ar_model, (y_train, 4), model_dir, "AR(4)"
    )


def process_arma_model(y_train, dataset_name):
    """Build ARMA(4,4) model"""
    model_dir = f"results/{dataset_name}/arma_model_p4_q4"
    return process_model_with_fit(
        y_train, dataset_name, build_arma_model, (y_train, 4, 4), model_dir, "ARMA(4,4)"
    )


def process_arima_model(y_train, dataset_name):
    """Build ARIMA(4,1,4) model"""
    config = (4, 1, 4)
    model = build_arima_model(y_train, order=config)
    print_model_summary(model, dataset_name, f"ARIMA{config}")

    model_dir = f"results/{dataset_name}/arima_model_p4_d1_q4"
    create_dir(model_dir)
    plot_residuals(
        model["residuals"],
        f"{dataset_name}: ARIMA{config} Residuals",
        f"{model_dir}/residuals.png",
    )

    plt.figure(figsize=(12, 6))
    t = np.arange(1, len(model["diff_y"]) + 1)
    plt.plot(t, model["diff_y"], "b-", linewidth=2)
    plt.xlabel("Time (k)")
    plt.ylabel("Differenced Value")
    plt.title(f"{dataset_name}: ARIMA{config} Differenced Series")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{model_dir}/differenced.png", dpi=100, bbox_inches="tight")
    plt.close()

    return model, config


def forecast_ar_static(model, y_train, y_test, dataset_name):
    """Static one-step ahead forecast for AR(4)"""
    p = 4
    y_full = np.concatenate([y_train, y_test])
    forecasts = []

    for i in range(len(y_test)):
        train_part = y_full[: len(y_train) + i]

        X = np.ones((len(train_part) - p, 1))
        for j in range(1, p + 1):
            X = np.column_stack([X, train_part[p - j : -j if j > 0 else None]])

        y_ar = train_part[p:]
        if len(y_ar) > 0:
            coeffs = np.linalg.lstsq(X, y_ar, rcond=None)[0]

            last_vals = np.concatenate([[1], train_part[-p:]])
            pred = np.dot(last_vals, coeffs)
            forecasts.append(pred)

    forecasts = np.array(forecasts)
    test_rmse = np.sqrt(np.mean((y_test - forecasts) ** 2))

    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train, forecasts])

    forecast_dir = f"results/{dataset_name}/ar_static_forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: AR(4) Static Forecast",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


def forecast_ar_dynamic(model, y_train, y_test, dataset_name):
    """Dynamic multi-step ahead forecast for AR(4)"""
    p = 4
    forecasts = list(y_train)

    for i in range(len(y_test)):
        train_part = np.array(forecasts)

        X = np.ones((len(train_part) - p, 1))
        for j in range(1, p + 1):
            X = np.column_stack([X, train_part[p - j : -j if j > 0 else None]])

        y_ar = train_part[p:]
        if len(y_ar) > 0:
            coeffs = np.linalg.lstsq(X, y_ar, rcond=None)[0]

            last_vals = np.concatenate([[1], train_part[-p:]])
            pred = np.dot(last_vals, coeffs)
            forecasts.append(pred)

    forecasts = np.array(forecasts[len(y_train) :])
    test_rmse = np.sqrt(np.mean((y_test - forecasts) ** 2))

    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train, forecasts])

    forecast_dir = f"results/{dataset_name}/ar_dynamic_forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: AR(4) Dynamic Forecast",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


def forecast_arma_static(model, y_train, y_test, dataset_name):
    """Static one-step ahead forecast for ARMA(4,4)"""
    p, q = 4, 4
    y_full = np.concatenate([y_train, y_test])
    forecasts = []

    for i in range(len(y_test)):
        train_part = y_full[: len(y_train) + i]

        X = np.ones((len(train_part) - max(p, q), 1))
        for j in range(1, p + 1):
            X = np.column_stack([X, train_part[max(p, q) - j : -j if j > 0 else None]])

        y_arma = train_part[max(p, q) :]
        if len(y_arma) > 0:
            coeffs = np.linalg.lstsq(X, y_arma, rcond=None)[0]

            last_vals = np.concatenate([[1], train_part[-p:]])
            pred = np.dot(last_vals[: len(coeffs)], coeffs)
            forecasts.append(pred)

    forecasts = np.array(forecasts)
    test_rmse = np.sqrt(np.mean((y_test - forecasts) ** 2))

    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train, forecasts])

    forecast_dir = f"results/{dataset_name}/arma_static_forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: ARMA(4,4) Static Forecast",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


def forecast_arma_dynamic(model, y_train, y_test, dataset_name):
    """Dynamic multi-step ahead forecast for ARMA(4,4)"""
    p, q = 4, 4
    forecasts = list(y_train)

    for i in range(len(y_test)):
        train_part = np.array(forecasts)

        X = np.ones((len(train_part) - max(p, q), 1))
        for j in range(1, p + 1):
            X = np.column_stack([X, train_part[max(p, q) - j : -j if j > 0 else None]])

        y_arma = train_part[max(p, q) :]
        if len(y_arma) > 0:
            coeffs = np.linalg.lstsq(X, y_arma, rcond=None)[0]

            last_vals = np.concatenate([[1], train_part[-p:]])
            pred = np.dot(last_vals[: len(coeffs)], coeffs)
            forecasts.append(pred)

    forecasts = np.array(forecasts[len(y_train) :])
    test_rmse = np.sqrt(np.mean((y_test - forecasts) ** 2))

    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train, forecasts])

    forecast_dir = f"results/{dataset_name}/arma_dynamic_forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: ARMA(4,4) Dynamic Forecast",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


def forecast_arima_static(model, config, y_train, y_test, dataset_name):
    """Static one-step ahead forecast for ARIMA"""
    p, d, q = config

    y_full = np.concatenate([y_train, y_test])

    forecasts = []
    for i in range(len(y_test)):
        train_part = y_full[: len(y_train) + i]

        diff_y = train_part.copy()
        for _ in range(d):
            diff_y = np.diff(diff_y)

        X = np.ones((len(diff_y) - max(p, q), 1))
        for j in range(1, p + 1):
            X = np.column_stack([X, diff_y[max(p, q) - j : -j if j > 0 else None]])

        y_ar = diff_y[max(p, q) :]
        coeffs = np.linalg.lstsq(X, y_ar, rcond=None)[0]

        last_vals = np.concatenate([[1], diff_y[-p:] if p > 0 else []])
        pred_diff = np.dot(
            last_vals[: min(len(coeffs), len(last_vals))],
            coeffs[: min(len(coeffs), len(last_vals))],
        )

        pred = train_part[-1] + pred_diff
        forecasts.append(pred)

    forecasts = np.array(forecasts)
    test_rmse = np.sqrt(np.mean((y_test - forecasts) ** 2))

    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train, forecasts])

    forecast_dir = f"results/{dataset_name}/arima_static_forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: ARIMA{config} Static Forecast",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


def forecast_arima_dynamic(model, config, y_train, y_test, dataset_name):
    """Dynamic multi-step forecast for ARIMA"""
    p, d, q = config

    forecasts = [y_train[-1]]

    for i in range(len(y_test)):
        train_part = np.concatenate([y_train, forecasts])

        diff_y = train_part.copy()
        for _ in range(d):
            diff_y = np.diff(diff_y)

        X = np.ones((len(diff_y) - max(p, q), 1))
        for j in range(1, p + 1):
            X = np.column_stack([X, diff_y[max(p, q) - j : -j if j > 0 else None]])

        y_ar = diff_y[max(p, q) :]
        coeffs = np.linalg.lstsq(X, y_ar, rcond=None)[0]

        last_vals = np.concatenate([[1], diff_y[-p:] if p > 0 else []])
        pred_diff = np.dot(
            last_vals[: min(len(coeffs), len(last_vals))],
            coeffs[: min(len(coeffs), len(last_vals))],
        )

        pred = train_part[-1] + pred_diff
        forecasts.append(pred)

    forecasts = np.array(forecasts[1:])
    test_rmse = np.sqrt(np.mean((y_test - forecasts) ** 2))

    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train, forecasts])

    forecast_dir = f"results/{dataset_name}/arima_dynamic_forecast"
    create_dir(forecast_dir)
    plot_trend(
        all_y,
        all_pred,
        f"{dataset_name}: ARIMA{config} Dynamic Forecast",
        f"{forecast_dir}/forecast.png",
    )

    return test_rmse


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10

    if not os.path.exists("results"):
        os.makedirs("results")

    print("\n" + "=" * 70)
    print("LAB 4 - VARIANT 9 - TREND AND ARIMA MODELING")
    print("=" * 70)

    datasets = {"IMPGE": "IMPGE.txt", "CURRNS": "CURRNS.txt"}

    for dataset_name, filename in datasets.items():
        y = load_data(filename)
        if y is None:
            continue

        print(f"\n\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"Sample size: {len(y)}")
        print(f"{'='*70}")

        train_size = len(y) - 8
        y_train = y[:train_size]
        y_test = y[train_size:]
        k_train = np.arange(1, len(y_train) + 1)

        print(f"\n--- TREND MODELS ---")
        best_trend = process_trend_models(y_train, k_train, dataset_name)
        print(
            f"\nBest trend: {best_trend['model_name']} (R² = {best_trend['r_squared']:.8f})"
        )

        print(f"\n--- AR(4) MODEL ---")
        ar_model = process_ar_model(y_train, dataset_name)

        print(f"\n--- ARMA(4,4) MODEL ---")
        arma_model = process_arma_model(y_train, dataset_name)

        print(f"\n--- ARIMA(4,1,4) MODEL ---")
        best_arima, best_config = process_arima_model(y_train, dataset_name)

        print(f"\n--- FORECASTS ---")
        trend_rmse = forecast_trend(best_trend, y_train, y_test, dataset_name)
        print(f"Trend forecast RMSE: {trend_rmse:.8f}")

        ar_static_rmse = forecast_ar_static(ar_model, y_train, y_test, dataset_name)
        print(f"AR(4) static RMSE: {ar_static_rmse:.8f}")

        ar_dynamic_rmse = forecast_ar_dynamic(ar_model, y_train, y_test, dataset_name)
        print(f"AR(4) dynamic RMSE: {ar_dynamic_rmse:.8f}")

        arma_static_rmse = forecast_arma_static(
            arma_model, y_train, y_test, dataset_name
        )
        print(f"ARMA(4,4) static RMSE: {arma_static_rmse:.8f}")

        arma_dynamic_rmse = forecast_arma_dynamic(
            arma_model, y_train, y_test, dataset_name
        )
        print(f"ARMA(4,4) dynamic RMSE: {arma_dynamic_rmse:.8f}")

        arima_static_rmse = forecast_arima_static(
            best_arima, best_config, y_train, y_test, dataset_name
        )
        print(f"ARIMA{best_config} static RMSE: {arima_static_rmse:.8f}")

        arima_dynamic_rmse = forecast_arima_dynamic(
            best_arima, best_config, y_train, y_test, dataset_name
        )
        print(f"ARIMA{best_config} dynamic RMSE: {arima_dynamic_rmse:.8f}")

        print(f"\nPlots saved to 'results/{dataset_name}' folder")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70 + "\n")
