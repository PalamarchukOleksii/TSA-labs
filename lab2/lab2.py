import numpy as np
import matplotlib.pyplot as plt
import os


# --- Генерація ряду v з нормального розподілу ---
def generate_noise_v(n: int = 100, seed: int = 42) -> np.ndarray:
    """
    Генерує псевдовипадковий нормально розподілений шум (аналог nrnd в EViews).

    :param n: Кількість точок у ряді
    :param seed: Початкове значення для генератора випадкових чисел (для відтворюваності)
    :return: Масив значень v
    """
    np.random.seed(seed)
    v = np.random.randn(n)
    return v


# --- Обчислення ряду y(k) за заданою формулою ---
def generate_series_y(v: np.ndarray,
                      a0: float, a1: float, a2: float, a3: float,
                      b1: float, b2: float, b3: float) -> np.ndarray:
    """
    Генерує часовий ряд y(k) за формулою:
    y(k) = a0 + a1*y(k-1) + a2*y(k-2) + a3*y(k-3) + v(k) + b1*v(k-1) + b2*v(k-2) + b3*v(k-3)

    :param v: Шумовий ряд (вхід)
    :return: Згенерований ряд y
    """
    n = len(v)
    y = np.zeros(n)

    # Початкові значення y(1..3)
    y[0:3] = v[0:3]

    # Генерація значень від k = 4 до 100
    for k in range(3, n):
        y[k] = (a0 +
                a1 * y[k - 1] + a2 * y[k - 2] + a3 * y[k - 3] +
                v[k] +
                b1 * v[k - 1] + b2 * v[k - 2] + b3 * v[k - 3])
    return y


# --- Візуалізація рядів ---
def plot_series(v: np.ndarray, y: np.ndarray, save_path: str = None):
    """
    Будує графіки рядів v і y на одній площині.

    :param v: Ряд шуму
    :param y: Генерований часовий ряд
    :param save_path: Шлях для збереження графіку (необов’язково)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(v, label='Випадковий шум v(k)')
    plt.plot(y, label='Сформований ряд y(k)', linestyle='--')
    plt.title('Часові ряди v(k) та y(k)')
    plt.xlabel('k')
    plt.ylabel('Значення')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Графік збережено у: {save_path}")
    else:
        plt.show()


# --- Зчитування файлів з папки data/ ---
def read_data_file(filename: str) -> np.ndarray:
    """
    Зчитує дані з файлу .txt (один стовпець чисел)

    :param filename: Ім’я файлу відносно папки data/
    :return: Масив значень з файлу
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "data", filename)
    try:
        data = np.loadtxt(filepath)
        print(f"Файл '{filename}' успішно зчитано.")
        return data
    except Exception as e:
        print(f"Помилка при зчитуванні файлу {filename}: {e}")
        return np.array([])

def simple_ma(y, N):
    n = len(y)
    ma = np.full(n, np.nan)
    for k in range(N-1, n):
        ma[k] = np.mean(y[k-N+1:k+1])
    return ma

def exp_ma(y, N):
    n = len(y)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (N + 1)
    
    weights = np.array([(1-alpha)**(N-i+1) for i in range(1, N+1)])
    weights_sum = np.sum(weights)
    
    for k in range(N-1, n):
        window = y[k-N+1:k+1]
        ema[k] = np.sum(window * weights) / weights_sum
    return ema

def acf(y, max_lag):
    n = len(y)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    variance = np.sum(y_centered ** 2)
    
    if variance == 0:
        return np.zeros(max_lag + 1)
    
    r = np.zeros(max_lag + 1)
    r[0] = 1.0
    
    for s in range(1, max_lag + 1):
        if s < n:
            covariance = np.sum(y_centered[:n-s] * y_centered[s:])
            r[s] = covariance / variance
    return r

def pacf(y, max_lag):
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
            numerator -= phi[k-1, j] * r[k-j]
        
        denominator = 1.0
        for j in range(1, k):
            denominator -= phi[k-1, j] * r[j]
        
        phi[k, k] = numerator / denominator if abs(denominator) > 1e-10 else 0
        phi_values[k] = phi[k, k]
        
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
    
    return phi_values

def plot_ma(y, ma_simple, ma_exp, N, filename, save_path):
    plt.figure(figsize=(14, 7))
    plt.plot(y, label='Вихідний ряд', alpha=0.5, linewidth=1)
    plt.plot(ma_simple, label=f'Просте КС (N={N})', linewidth=2)
    plt.plot(ma_exp, label=f'Експоненційне КС (N={N})', linewidth=2, linestyle='--')
    plt.title(f'{filename} з ковзними середніми (N={N})')
    plt.xlabel('k')
    plt.ylabel('Значення')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Збережено: {save_path}")

def plot_correlations(acf_vals, pacf_vals, filename, save_path):
    lags = np.arange(len(pacf_vals))
    ci = 1.96 / np.sqrt(len(pacf_vals))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].stem(lags, acf_vals, basefmt=' ')
    axes[0].axhline(y=0, color='k', linewidth=0.8)
    axes[0].axhline(y=ci, color='r', linestyle='--', label='95% ДІ')
    axes[0].axhline(y=-ci, color='r', linestyle='--')
    axes[0].set_title(f'АКФ - {filename}')
    axes[0].set_xlabel('Лаг')
    axes[0].set_ylabel('АКФ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(lags, pacf_vals, basefmt=' ')
    axes[1].axhline(y=0, color='k', linewidth=0.8)
    axes[1].axhline(y=ci, color='r', linestyle='--', label='95% ДІ')
    axes[1].axhline(y=-ci, color='r', linestyle='--')
    axes[1].set_title(f'ЧАКФ - {filename}')
    axes[1].set_xlabel('Лаг')
    axes[1].set_ylabel('ЧАКФ')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Збережено: {save_path}")

def process_file(filename, window_size):
    y = read_data_file(filename)
    base_name = os.path.splitext(filename)[0]

    ma_simple = simple_ma(y, window_size)
    ma_exp = exp_ma(y, window_size)

    max_lag = min(30, len(y) // 4)
    acf_vals = acf(y, max_lag)
    pacf_vals = pacf(y, max_lag)
    
    print(f"\nПерші 10 значень ЧАКФ:")
    for i in range(min(10, len(pacf_vals))):
        print(f"  Лаг {i}: {pacf_vals[i]:.4f}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    plot_ma(
        y,
        ma_simple,
        ma_exp,
        window_size,
        base_name,
        os.path.join(results_dir, f"{base_name}_ma_N{window_size}.png")
    )

    plot_correlations(
        acf_vals,
        pacf_vals,
        base_name,
        os.path.join(results_dir, f"{base_name}_pacf.png")
    )

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Коефіцієнти з табл. A.1 (для бригади №1)
    a0 = 0
    a1 = 0.22
    a2 = -0.18
    a3 = 0.08
    b1 = 0.5
    b2 = 0.25
    b3 = 0.25

    v = generate_noise_v(n=100)
    y = generate_series_y(v, a0, a1, a2, a3, b1, b2, b3)

    output_path = os.path.join(results_dir, "v_y_plot.png")
    plot_series(v, y, save_path=output_path)
    
    process_file("rts1.txt", 5)
    process_file("rts1.txt", 10)

    process_file("1996rts1.txt", 5)
    process_file("1996rts1.txt", 10)

if __name__ == "__main__":
    main()
