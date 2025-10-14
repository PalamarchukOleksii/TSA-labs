import numpy as np
import matplotlib.pyplot as plt
import os


def generate_noise_v(n: int = 100, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    v = np.random.randn(n)
    return v


def generate_series_y(
    v: np.ndarray,
    a0: float,
    a1: float,
    a2: float,
    a3: float,
    b1: float,
    b2: float,
    b3: float,
) -> np.ndarray:
    n = len(v)
    y = np.zeros(n)
    y[0:3] = v[0:3]

    for k in range(3, n):
        y[k] = (
            a0
            + a1 * y[k - 1]
            + a2 * y[k - 2]
            + a3 * y[k - 3]
            + v[k]
            + b1 * v[k - 1]
            + b2 * v[k - 2]
            + b3 * v[k - 3]
        )
    return y


def plot_series(v: np.ndarray, y: np.ndarray, save_path: str = None) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(v, label="Random noise v(k)", alpha=0.7)
    plt.plot(y, label="Generated series y(k)", linestyle="--", linewidth=2)
    plt.title("Time Series v(k) and y(k)")
    plt.xlabel("Time index k")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def read_data_file(filename: str) -> np.ndarray:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "data", filename)
    try:
        data = np.loadtxt(filepath)
        print(f"File '{filename}' successfully loaded. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return np.array([])


def simple_ma(y: np.ndarray, N: int) -> np.ndarray:
    n = len(y)
    ma = np.full(n, np.nan)

    for k in range(N - 1, n):
        ma[k] = np.mean(y[k - N + 1 : k + 1])
    return ma


def exp_ma(y: np.ndarray, N: int) -> np.ndarray:
    n = len(y)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (N + 1)

    weights = np.array([(1 - alpha) ** (N - i + 1) for i in range(1, N + 1)])
    weights_sum = np.sum(weights)

    for k in range(N - 1, n):
        window = y[k - N + 1 : k + 1]
        ema[k] = np.sum(window * weights) / weights_sum
    return ema


def acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(y)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    variance = np.sum(y_centered**2)

    if variance == 0:
        return np.zeros(max_lag + 1)

    r = np.zeros(max_lag + 1)
    r[0] = 1.0

    for s in range(1, max_lag + 1):
        if s < n:
            covariance = np.sum(y_centered[: n - s] * y_centered[s:])
            r[s] = covariance / variance
    return r


def pacf(y: np.ndarray, max_lag: int) -> np.ndarray:
    r = acf(y, max_lag)
    phi_values = np.zeros(max_lag + 1)
    phi_values[0] = 1.0

    if max_lag >= 1:
        phi_values[1] = r[1]

    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = r[1]

    for k in range(2, max_lag + 1):
        numerator = r[k]
        for j in range(1, k):
            numerator -= phi[k - 1, j] * r[k - j]

        denominator = 1.0
        for j in range(1, k):
            denominator -= phi[k - 1, j] * r[j]

        phi[k, k] = numerator / denominator if abs(denominator) > 1e-10 else 0
        phi_values[k] = phi[k, k]

        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return phi_values


def plot_ma(
    y: np.ndarray,
    ma_simple: np.ndarray,
    ma_exp: np.ndarray,
    N: int,
    filename: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(y, label="Original series", alpha=0.5, linewidth=1, color="gray")
    plt.plot(ma_simple, label=f"Simple MA (N={N})", linewidth=2, color="blue")
    plt.plot(
        ma_exp,
        label=f"Exponential MA (N={N})",
        linewidth=2,
        linestyle="--",
        color="red",
    )
    plt.title(f"{filename} with Moving Averages (N={N})")
    plt.xlabel("Time index k")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlations(
    acf_vals: np.ndarray, pacf_vals: np.ndarray, filename: str, save_path: str
) -> None:
    lags = np.arange(len(pacf_vals))
    ci = 1.96 / np.sqrt(len(pacf_vals))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].stem(lags, acf_vals, basefmt=" ")
    axes[0].axhline(y=0, color="k", linewidth=0.8)
    axes[0].axhline(y=ci, color="r", linestyle="--", label="95% CI", alpha=0.7)
    axes[0].axhline(y=-ci, color="r", linestyle="--", alpha=0.7)
    axes[0].set_title(f"Autocorrelation Function (ACF) - {filename}")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(lags, pacf_vals, basefmt=" ")
    axes[1].axhline(y=0, color="k", linewidth=0.8)
    axes[1].axhline(y=ci, color="r", linestyle="--", label="95% CI", alpha=0.7)
    axes[1].axhline(y=-ci, color="r", linestyle="--", alpha=0.7)
    axes[1].set_title(f"Partial Autocorrelation Function (PACF) - {filename}")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("PACF")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def process_file(filename: str, window_size: int) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {filename} with window size N={window_size}")
    print(f"{'='*60}")

    y = read_data_file(filename)

    if len(y) == 0:
        print(f"No data to process for {filename}")
        return

    base_name = os.path.splitext(filename)[0]

    ma_simple = simple_ma(y, window_size)
    ma_exp = exp_ma(y, window_size)

    max_lag = min(30, len(y) // 4)
    acf_vals = acf(y, max_lag)
    pacf_vals = pacf(y, max_lag)

    print(f"\nFirst 10 PACF values:")
    for i in range(len(pacf_vals)):
        significance = "***" if abs(pacf_vals[i]) > 1.96 / np.sqrt(len(y)) else ""
        print(f"  Lag {i}: {pacf_vals[i]:7.4f} {significance}")

    print(f"\nStatistics for {filename}:")
    print(f"  Series length: {len(y)}")
    print(f"  Mean: {np.mean(y):.4f}")
    print(f"  Std Dev: {np.std(y):.4f}")
    print(f"  Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    plot_ma(
        y,
        ma_simple,
        ma_exp,
        window_size,
        base_name,
        os.path.join(results_dir, f"{base_name}_ma_N{window_size}.png"),
    )

    plot_correlations(
        acf_vals,
        pacf_vals,
        base_name,
        os.path.join(results_dir, f"{base_name}_correlations.png"),
    )


def main() -> None:
    print("\n" + "=" * 60)
    print("TIME SERIES ANALYSIS WITH ARMA MODELING")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    a0 = 0
    a1 = 0.22
    a2 = -0.18
    a3 = 0.08
    b1 = 0.5
    b2 = 0.25
    b3 = 0.25

    print("\n--- PART 1: Synthetic ARMA(3,3) Series Generation ---")
    print(f"Model: y(k) = {a0} + {a1}*y(k-1) + {a2}*y(k-2) + {a3}*y(k-3)")
    print(f"       + v(k) + {b1}*v(k-1) + {b2}*v(k-2) + {b3}*v(k-3)")

    v = generate_noise_v(n=100)
    print(
        f"Generated white noise v(k): n=100, mean={np.mean(v):.4f}, std={np.std(v):.4f}"
    )

    y = generate_series_y(v, a0, a1, a2, a3, b1, b2, b3)
    print(f"Generated ARMA series y(k): mean={np.mean(y):.4f}, std={np.std(y):.4f}")

    output_path = os.path.join(results_dir, "v_y_plot.png")
    plot_series(v, y, save_path=output_path)

    print("\n--- PART 2: Real Time Series Analysis ---")

    process_file("rts1.txt", 5)
    process_file("rts1.txt", 10)
    process_file("1996rts1.txt", 5)
    process_file("1996rts1.txt", 10)

    print("\n" + "=" * 60)
    print("Analysis complete! Check the 'results/' folder for plots.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
