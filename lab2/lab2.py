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
    filepath = os.path.join("data", filename)
    try:
        data = np.loadtxt(filepath)
        print(f"Файл '{filename}' успішно зчитано.")
        return data
    except Exception as e:
        print(f"Помилка при зчитуванні файлу {filename}: {e}")
        return np.array([])


# --- Основна функція ---
def main():
    # Коефіцієнти з табл. A.1 (для бригади №1)
    a0 = 0
    a1 = 0.22
    a2 = -0.18
    a3 = 0.08
    b1 = 0.5
    b2 = 0.25
    b3 = 0.25

    # 1.2: Генерація ряду v
    v = generate_noise_v(n=100)

    # 1.3: Генерація ряду y(k)
    y = generate_series_y(v, a0, a1, a2, a3, b1, b2, b3)

    # Побудова графіка та збереження у results/
    output_path = os.path.join("results", "v_y_plot.png")
    plot_series(v, y, save_path=output_path)

    # [Додатково] Приклад зчитування файлів із data/
    # rts1 = read_data_file("rts1.txt")
    # rts2 = read_data_file("1996rts1.txt")

    # [Опційно] Можна також побудувати графік завантажених даних:
    # plot_series(rts1, rts2)


if __name__ == "__main__":
    main()
