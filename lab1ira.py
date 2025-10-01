import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================== ПАРАМЕТРИ ==================
MODE = "file"         # "auto" = генерувати дані, "file" = читати з файлу
DATA_FOLDER = "data"  # папка для файлів, якщо MODE = "file"

P, Q = 3, 3           # порядок ARMA
OBS = 500             # кількість спостережень
ARMA_ALL = True       # рахувати всі моделі (True/False)

# Фіксовані коефіцієнти
# a0, a1, a2, a3, b1, b2, b3
COEFFS = [0, 0.22, -0.18, 0.08, 0.5, 0.25, 0.25]

# ==============================================

# Функція для отримання значень (без вводу від користувача)
def read_user_input():
    folder = MODE
    arma_all = ARMA_ALL
    arma = [P, Q]
    coeffs = COEFFS
    return folder, arma, coeffs, arma_all

# Функція для читання файлів
def read_file(data_dir = "data"):
    y_file = os.path.join(data_dir, 'y.txt')
    v_file = os.path.join(data_dir, 'v.txt')
    
    if not os.path.exists(y_file) or not os.path.exists(v_file):
        raise FileNotFoundError(f"Required files not found: {y_file}, {v_file}")
    
    y = np.loadtxt(y_file)
    v = np.loadtxt(v_file)
    
    return y, v

# Функція, що генерує білий шум
def gen_noise(num_samples=100, mean=0, std=1):
    white_noise = np.random.normal(mean, std, num_samples)
    return white_noise

# КОЕФІЦІЄНТ БЕТА
BETTA = 2

# S - сума квадратів похибок
def S(y_true, y_pred):
    e_squared = (np.array(y_true) - np.array(y_pred))**2
    return e_squared.sum()

# R^2 - коефіцієнт детермінації
def R_squared(y_true, y_pred):
    var_pred = np.array(y_pred).var()
    var_true = np.array(y_true).var()
    return var_pred / var_true

# Критерій Акаіке
def IKA(y_true, y_pred, n):
    epsilon = 1e-10
    return len(y_true) * np.log(S(y_true, y_pred) + epsilon) + 2*n

# Генерація ARMA
def arma(p, q, coeffs:dict, t:int, e:list or tuple):
    y = np.random.rand(t)
    start_point = max([p, q])

    for i in range(start_point, t):
        y[i] = coeffs['a0']
        for j in range(1, p + 1):
            y[i] += coeffs['a' + str(j)]*y[i - j]
        for j in range(1, q + 1):
            y[i] += coeffs['b' + str(j)]*e[i - j]
    return y

# МНК
def mnk(y, X):
    res = np.array([])
    y = np.array(y)
    try:
        res = (np.linalg.inv(X.T@X)) @ X.T @ y[int(y.shape[0] - X.shape[0]):]
    except np.linalg.LinAlgError:
        print('Матриця Х виявилась виродженою')
    except Exception as er:
        print(er)
        print(X.shape, y.shape)
    finally:
        return res

# Генерація матриці X
def get_x(y, v, p, q, obs):
    start_point = max([p, q])
    X = [[1 for _ in range(obs - start_point)]]
    for i in range(1, p + 1):
        X.append(y[start_point - i:obs - i])
    for i in range(1, q + 1):
        X.append(v[start_point - i:obs - i])
    return np.array(X).T

# РМНК
def rmnk(y, X, beta):
    x_row, x_col = X.shape[0], X.shape[1]
    p0 = np.identity(x_col) * beta
    coeffs = np.matrix(np.zeros(x_col)).T
    for i in range(x_row):
        row = np.matrix(X[i])
        p1 = p0 - (p0 @ row.T @ row @ p0) / (1 + (row @ p0 @ row.T).tolist()[0][0])
        coeffs = coeffs + p1 @ row.T * (y[i] - (row @ coeffs).tolist()[0][0])
        p0 = p1
    return np.squeeze(np.asarray(coeffs))

# Підрахунок метрик
def count_metrics(y_true, y_pred_mnk, y_pred_rmnk, p, q):
    start_point = max([p, q])
    n = p + q + 1
    result = []

    for y_pred in [y_pred_mnk, y_pred_rmnk]:
        result.append(S(y_true[start_point:], y_pred))
        result.append(R_squared(y_true[start_point:], y_pred))
        result.append(IKA(y_true[start_point:], y_pred, n))

    return result

# Побудова графіків
def plot_all_we_need_to_plot(p, q, teta, obs, y, v, arma_all=False, result_metrics=None, labels=None):
    best_params = []
    for i in range(p + 1):
        best_params.append(teta['a' + str(i)])
    for i in range(1, q + 1):
        best_params.append(teta['b' + str(i)])

    m = len(best_params)
    n = obs - 10

    mnk_params = []
    rmnk_params = []

    for k in range(10, obs):
        X = get_x(y[:k], v[:k], p, q, k)
        mnk_params.append(mnk(y[:k], X))
        rmnk_params.append(rmnk(y[:k], X, BETTA))

    mnk_params = np.array(mnk_params)
    rmnk_params = np.array(rmnk_params)

    param_titles = ['a' + str(i) for i in range(p + 1)] + ['b' + str(i) for i in range(1, q + 1)]

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(m):
        axs[int(i / 3), i % 3].plot(range(n), mnk_params[:, i], label='МНК', color='red')
        axs[int(i / 3), i % 3].plot(range(n), rmnk_params[:, i], label='РМНК', color='blue')
        axs[int(i / 3), i % 3].plot(range(n), np.ones(n) * best_params[i], label='Еталонні значення', color='green', linestyle='--')
        axs[int(i / 3), i % 3].set_title(param_titles[i])
        axs[int(i / 3), i % 3].set_xlabel('k')
        axs[int(i / 3), i % 3].legend()
    plt.tight_layout()
    plt.savefig("params_plot.png")
    plt.close()

    if arma_all:
        metrics = np.array(result_metrics)
        titles = ['S', 'R^2', 'IKA']
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            axs[i].plot(labels, metrics[:, i], label='МНК', color='blue', marker='o')
            axs[i].plot(labels, metrics[:, i + 3], label='РМНК', color='yellow', marker='o')
            axs[i].set_title(titles[i])
            for tick in axs[i].get_xticklabels():
                tick.set_rotation(30)
            axs[i].legend()
        plt.tight_layout()
        plt.savefig("metrics_plot.png")
        plt.close()

# Головна функція
def main():
    folder, (p, q), coeffs, arma_all = read_user_input()
    teta = dict(zip(['a' + str(i) for i in range(p + 1)] + ['b' + str(i) for i in range(1, q + 1)], coeffs))

    if folder == 'auto':
        obs = OBS
        v = gen_noise(obs)
        y = arma(p, q, teta, obs, v)
    elif folder == 'file':
        current_directory = os.path.dirname(os.path.abspath(__file__))
        y, v = read_file(os.path.join(current_directory, DATA_FOLDER))
        obs = min([len(v), len(y)])
    else:
        raise ValueError("MODE має бути 'auto' або 'file'")

    labels = []
    if arma_all:
        result_metrics = []
        for i in range(1, p + 1):
            for j in range(1, q + 1):
                X = get_x(y, v, i, j, obs)
                mnk_output = mnk(y, X)
                rmnk_output = rmnk(y, X, BETTA)
                mnk_pred = X @ np.array(mnk_output).reshape(-1, 1)
                rmnk_pred = X @ np.array(rmnk_output).reshape(-1, 1)
                result_metrics.append(count_metrics(y, mnk_pred, rmnk_pred, i, j))
                labels.append(f'АРКС({i}, {j})')
        df = pd.DataFrame(dict(zip(labels, result_metrics)))
    else:
        X = get_x(y, v, p, q, obs)
        mnk_output = mnk(y, X)
        rmnk_output = rmnk(y, X, BETTA)
        mnk_pred = X @ np.array(mnk_output).reshape(-1, 1)
        rmnk_pred = X @ np.array(rmnk_output).reshape(-1, 1)
        result_metrics = count_metrics(y, mnk_pred, rmnk_pred, p, q)
        df = pd.DataFrame({f"АРКС({p}, {q})": result_metrics})

    df['Метод'] = ['МНК', 'МНК', 'МНК', 'РМНК', 'РМНК', 'РМНК']
    df['Метрика'] = ['S', 'R2', 'IKA', 'S', 'R2', 'IKA']
    df.set_index(['Метод', 'Метрика'], inplace=True)
    pd.options.display.float_format = '{:.3f}'.format
    print(df.transpose())

    plot_all_we_need_to_plot(p, q, teta, obs, y, v, arma_all=arma_all, result_metrics=result_metrics, labels=labels)

if __name__ == '__main__':
    main()
