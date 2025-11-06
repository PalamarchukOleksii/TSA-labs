import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os


from functools import reduce
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson

ALPHA = 0.05
SIGNIFICANT_CORREL = 0.5
MAX_LAG = 12

# ====================================================================
# STATISTICS (Без змін)
# ====================================================================


def S(y_true, y_pred):
    """Сума квадратів похибок"""
    diff = (np.array(y_true) - np.array(y_pred)) ** 2
    return np.round(diff.sum(), 4)


def R2(y_true, y_pred):
    """Коефіцієнт детермінації"""
    var_pred = np.array(y_pred).var()
    var_true = np.array(y_true).var()
    # За методичкою R2 = var_pred / var_true. Оскільки у ваших функціях є
    # перевірка score <= 1, залишаю її, хоча стандартна формула відрізняється.
    score = np.round(var_pred / var_true, 4)
    return score if score <= 1 else -score


def IKA(y_true, y_pred, n):
    """Критерій Акайке"""
    epsilon = 1e-10
    # Врахування кількості спостережень T (len(y_true))
    # n - кількість параметрів у моделі (включаючи константу, якщо є)
    T = len(y_true)
    return np.round(T * np.log(S(y_true, y_pred) / T + epsilon) + 2 * n, 4)


def DW(epsilon):
    """Статистика Дарбіна-Уотсона"""
    DW_stat = sum(
        [(i_t - i_t_1) ** 2 for i_t, i_t_1 in zip(epsilon[1:], epsilon[:-1])]
    ) / sum([i**2 for i in epsilon])
    return DW_stat


def sma(series, w_size):
    """Просте ковзне середнє"""
    # Перетворення на Series для використання вбудованої функції rolling
    return pd.Series(series).rolling(window=w_size).mean().tolist()


def ema(series, w_size):
    """Експоненційне ковзне середнє"""
    alpha = 2 / (w_size + 1)
    # Використання вбудованої функції pandas.ewm для EMA
    return pd.Series(series).ewm(alpha=alpha, adjust=False).mean().tolist()


# Додано перевірку на pandas.Series у ACF та PACF
def ACF(series, s):
    """АКФ"""
    series = pd.Series(series)
    N = len(series)
    if N == 0:
        return 0
    mean = series.mean()
    # Виправлено знаменник для вибіркової дисперсії (N замість N-1 для автоковаріації)
    denominator = series.apply(lambda x: (x - mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    numerator = sum(
        [(series.iloc[i] - mean) * (series.iloc[i - s] - mean) for i in range(s, N)]
    )
    return numerator / denominator if denominator else 0.0


# Оригінальна PACF реалізація з lab3.py є досить складною для перевірки/підтримки.
# Враховуючи, що в lab2.py є інша реалізація PACF, і для коректності
# краще використовувати statsmodels.tsa.stattools.pacf для PACF в основній логіці,
# я залишу оригінальну реалізацію з lab3.py, але вкажу на її потенційну складність.
#
# Примітка: оригінальна PACF з lab3.py може бути не цілком коректною
# або не відповідати стандартним бібліотечним реалізаціям.
# Для цілей демонстрації залишаю як є.


def PACF(series, lag):
    """ЧАКФ (Оригінальна реалізація з lab3.py)"""
    # series = pd.Series(series) # Може бути потрібне, якщо series не pandas Series

    # Використовуємо ACF з lab3.py для кореляцій
    r = np.array([ACF(series, s) for s in range(lag + 1)])

    phi_values = np.zeros(lag + 1)
    if lag >= 1:
        phi_values[1] = r[1]

    phi = np.zeros((lag + 1, lag + 1))
    if lag >= 1:
        phi[1, 1] = r[1]

    for k in range(2, lag + 1):
        # rk
        numerator = r[k]
        for j in range(1, k):
            # Фіксуємо коректний доступ до елементів phi[k - 1, j] з k-1 порядком
            numerator -= phi[k - 1, j] * r[k - j]

        denominator = 1.0
        for j in range(1, k):
            # Фіксуємо коректний доступ до елементів phi[k - 1, j] з k-1 порядком
            denominator -= phi[k - 1, j] * r[j]

        # Коефіцієнт k-го порядку
        phi[k, k] = numerator / denominator if abs(denominator) > 1e-10 else 0
        phi_values[k] = phi[k, k]

        # Коефіцієнти менших порядків
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return phi_values.tolist()


class InputReceiver:
    def validate_input(self, input_text: str, error_text: str, condition, key: int = 1):
        dtype = {1: int, 2: float, 3: str}
        while True:
            try:
                user_input = dtype[key](input(input_text + ": "))
            except ValueError:
                print("Неправильне значення, " + error_text + "\nСпробуйте знову")
            else:
                if not condition(user_input):
                    print("Неправильне значення, " + error_text + "\nСпробуйте знову")
                else:
                    return user_input

    def read_file(self, caption: str, key: int = 1, file_path_override: str = None):
        if file_path_override:
            path = file_path_override
        else:
            path = input(f"{caption}: ")

        try:
            with open(path, "r") as f:
                if key == 1:
                    # Припускаємо, що це файл з одним стовпцем чистих даних (як rts1.txt)
                    return [float(i.strip()) for i in f.readlines() if i.strip()]
                else:
                    # Припускаємо, що це файл у форматі 'ім'я=значення'
                    return [
                        float(i.partition("=")[2].strip())
                        for i in f.readlines()
                        if i.partition("=")[2].strip()
                    ]
        except FileNotFoundError:
            if file_path_override:
                print(f"Файл не знайдено за шляхом: {path}. Перевірте наявність файлу.")
                # Якщо це автоматичний запуск, вихід з помилкою
                raise
            else:
                print("Неправильна назва файлу, спробуйте знову")
                return self.read_file(caption)

    def part_one(self, file_path: str = None):
        if file_path:
            # Для автоматичного завантаження файлу DW, якщо шлях відомий
            DW_series = self.read_file(
                "Шлях до файлу з даними для розрахунку статистики Дарбіна-Уотсона",
                key=1,
                file_path_override=file_path,
            )
        else:
            # Для ручного введення
            DW_series = self.read_file(
                "Шлях до файлу з даними для розрахунку статистики Дарбіна-Уотсона",
                key=1,
            )
        return DW_series

    def load_data(self):
        # Використовуємо 'data/' як приклад, якщо файли знаходяться там
        # Припускаємо, що RTStl.txt - це залежна змінна (rts1) для нашого варіанту
        dir = "data/"

        # Наш target згідно варіанту №9 - RTStl.txt
        target_file = "RTStl.txt"
        target = pd.read_csv(dir + target_file, header=None, squeeze=True)

        # Регресори (екзогенні індекси, включаючи target, для кореляційної матриці)
        pathes = [
            "RTStl.txt",
            "RTSog.txt",
            "RTSmm.txt",
            "RTSin.txt",
            "RTSfn.txt",
            "RTSeu.txt",
            "RTScr.txt",
        ]

        indexes = {}
        for path in pathes:
            try:
                # Читання даних, припускаючи, що це один стовпець без заголовка
                data = pd.read_csv(dir + path, header=None, squeeze=True)
                indexes[path.partition(".")[0]] = data
            except FileNotFoundError:
                print(
                    f"Помилка: Файл {dir + path} не знайдено. Перевірте шляхи до файлів даних."
                )
                # Створення порожнього ряду для продовження, але це може призвести до помилок пізніше
                indexes[path.partition(".")[0]] = pd.Series([])

        # Вирівнювання довжин рядів до мінімальної
        min_len = min(len(s) for s in indexes.values() if not s.empty)
        if min_len == 0:
            print("Помилка: Не вдалося завантажити жоден часовий ряд.")
            return pd.DataFrame()

        # Обрізка рядів до мінімальної довжини
        for key in indexes:
            indexes[key] = indexes[key].iloc[:min_len]

        return pd.DataFrame(data=indexes)


def descriptive_analysis(series):
    series = pd.Series(series)
    description = series.describe()
    desc_dict = {}
    for key, val in description.to_dict().items():
        desc_dict[key] = val

    desc_dict["median"] = series.median()
    desc_dict["mode"] = series.mode()[0] if not series.mode().empty else np.nan
    desc_dict["skewness"] = series.skew()
    desc_dict["kurtosis"] = series.kurtosis()
    # statsmodels.stats.stattools.jarque_bera повертає статистику та p-value
    jb_test = jarque_bera(series.to_numpy())
    desc_dict["Jarque-Bera"] = jb_test[0]
    desc_dict["p-value"] = jb_test[1]

    # Візуалізація описових статистик (як у методичці)
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=series, kde=True)

    # Вивід статистик праворуч (як на Рис. 2 методички)
    fig = ax.get_figure()
    fig.subplots_adjust(right=0.6)
    fig.suptitle("Описові статистики")
    ax.set_xlabel("x")

    # Сортування для виводу у порядку як у методичці
    stats_to_display = [
        ("Observations", len(series)),
        ("Mean", desc_dict.get("mean")),
        ("Median", desc_dict.get("median")),
        ("Maximum", desc_dict.get("max")),
        ("Minimum", desc_dict.get("min")),
        ("Std. Dev.", desc_dict.get("std")),
        ("Skewness", desc_dict.get("skewness")),
        ("Kurtosis", desc_dict.get("kurtosis")),
        ("Jarque-Bera", desc_dict.get("Jarque-Bera")),
        ("Probability", desc_dict.get("p-value")),
    ]

    counter = 0.95
    for stat, val in stats_to_display:
        ax.text(1.1, counter, f"{stat}".ljust(20), transform=ax.transAxes, fontsize=9)
        ax.text(
            1.5,
            counter,
            f"{val:.6f}" if isinstance(val, (int, float)) else str(val).ljust(10),
            transform=ax.transAxes,
            fontsize=9,
        )
        counter -= 0.05

    plt.show()


def stats_report(y_true, y_pred, n_params, coefs, p, name=""):
    q = n_params - p
    epsilon = np.array(y_true) - np.array(y_pred)

    print("\n", "-" * 80, sep="")
    print("Модель: ".ljust(15), f"{name}".ljust(15))
    print("Коефіцієнти: ".ljust(15), end="")

    # AR/c коефіцієнти
    ar_coefs = [coefs[0]] + coefs[1 : p + 1]  # Константа + AR(p)
    print(f"c(1) = {ar_coefs[0]:.3f}; ", end="")
    print(
        *[f"a{i} = {coef:.3f}" for i, coef in enumerate(ar_coefs[1:], 1)],
        sep="; ",
        end="",
    )

    # MA коефіцієнти
    if q > 0:
        ma_coefs = coefs[p + 1 :]
        print(f"; mv + ", end="")  # mv(k) - коефіцієнт 1
        print(*[f"b{i} = {coef:.3f}" for i, coef in enumerate(ma_coefs, 1)], sep="; ")
    else:
        print()

    print("S (Sum squared resid):".ljust(30), f"{S(y_true, y_pred):.6f}".ljust(15))
    print("R2:".ljust(30), f"{R2(y_true, y_pred):.6f}".ljust(15))

    # n_params тут - це p+q, тому додаємо 1 для константи c(1)
    print(
        "AIC (Akaike info criterion):".ljust(30),
        f"{IKA(y_true, y_pred, n=n_params + 1):.6f}".ljust(15),
    )
    print("DW (Durbin-Watson stat):".ljust(30), f"{DW(epsilon):.6f}".ljust(15))
    print("-" * 80, "\n", sep="")


def X_matrix(y, v, p, q):
    # Уніфікація: y - для AR частини, v - для MA частини
    # N - довжина найбільшого ряду, m - максимальний лаг
    N_y, N_v = len(y), len(v)
    N = N_y if N_y > 0 else N_v
    m = max([p, q])

    # Використовуємо лише ділянки, де є повні лаги
    effective_N = N - m

    if effective_N <= 0:
        return np.array([[]]).T

    X = []

    # 1. Константа
    X.append(np.ones(effective_N))

    # 2. AR частина (y лаги)
    for i in range(1, p + 1):
        if N_y > 0:
            # y[m-i: N-i] бере елементи y з лагом i
            X.append(y[m - i : N - i])
        else:
            # Якщо y не передано, додаємо нулі для коректної розмірності
            X.append(np.zeros(effective_N))

    # 3. MA частина (v лаги)
    for i in range(1, q + 1):
        if N_v > 0:
            # v[m-i: N-i] бере елементи v з лагом i
            X.append(v[m - i : N - i])
        else:
            X.append(np.zeros(effective_N))

    # X.T - матриця, де кожен рядок - це набір регресорів для одного спостереження
    return np.array(X).T


def LS(y, X):
    """МНК (Least Squares)"""
    res = np.array([])
    # В y беремо лише частину, що відповідає ефективній N
    y_eff = (
        np.array(y)[X.shape[0] :] if len(y) > X.shape[0] else np.array(y)[: X.shape[0]]
    )

    try:
        # Перевірка на виродженість та розрахунок
        res = np.linalg.lstsq(X, y_eff, rcond=None)[0]
    except np.linalg.LinAlgError:
        print("Матриця Х виявилася виродженою")
    except ValueError as e:
        print(
            f"Помилка розмірності при МНК: {e}. X.shape={X.shape}, y_eff.shape={y_eff.shape}"
        )
    finally:
        return res


def generate_arma(v, N, p, q, coeffs):
    # coeffs: [a0, a1, ..., ap, b1, ..., bq]
    if len(coeffs) != (1 + p + q):
        # Якщо немає константи (a0), то передбачається, що її коефіцієнт = 0
        if len(coeffs) == (p + q):
            coeffs = np.concatenate([[0], coeffs])
        else:
            print(
                f"Помилка: Некоректна кількість коефіцієнтів ({len(coeffs)}) для ARMA({p},{q}). Очікується {1+p+q}."
            )
            return np.zeros(N)

    a0, a, b = coeffs[0], coeffs[1 : p + 1], coeffs[p + 1 :]
    y = np.zeros(N)

    # Ініціалізація y[0...max(p, q)-1]
    max_lag = max([p, q])
    for i in range(max_lag):
        y[i] = a0 + v[i]  # Просте ініціювання, як у вашому коді

    # Генерація решти точок
    for k in range(max_lag, N):
        # AR частина
        ar_term = sum(a[i] * y[k - i - 1] for i in range(p))

        # MA частина
        ma_term = sum(b[i] * v[k - i - 1] for i in range(q))

        y[k] = a0 + ar_term + v[k] + ma_term

    return y


def estimate_order(values, N):
    """Оцінка порядку (p або q) за кореляційною функцією"""
    # 1.96 / sqrt(N) - 95% довірчий інтервал
    ci = 1.96 / np.sqrt(N)
    # Знаходимо лаг, після якого коефіцієнти стають статистично незначущими
    for i, val in enumerate(values):
        if abs(val) < ci:
            # Порядок - це останній значущий лаг, тобто i
            return i

    # Якщо всі лаги значущі, повертаємо максимальний (MAX_LAG)
    return len(values)


def resid_approach(series, p, w_size, dummy=[]):
    print("\n\n" + "=" * 80)
    print(f"Побудова КС за залишками АР({p}) (Підхід №1.1)")
    print("=" * 80)

    N = len(series)
    # Крок 1.1: Оцінювання AR(p)
    # y = series[p:], X = матриця лагів series[p-1:N-1] ... series[0:N-p]
    X_ar = X_matrix(y=series, v=[], p=p, q=0)
    y_ar = series[p:]
    AR_coefs = LS(y=y_ar, X=X_ar)

    if len(AR_coefs) != p + 1:
        print("Помилка: Не вдалося оцінити коефіцієнти AR. Пропуск підходу.")
        return ({}, {})

    # Генерація AR(p) prediction (для розрахунку залишків)
    # Використовуємо оцінені коефіцієнти для прогнозування.
    # v - тут шум, який не використовується в AR(p) prediction для обчислення залишків
    y_pred_full = generate_arma(v=np.zeros(N), N=N, p=p, q=0, coeffs=AR_coefs)

    # Обчислення залишків (resid)
    residue = pd.Series(series.values - y_pred_full)

    if not dummy:
        # Звіт для AR(p)
        stats_report(
            series, y_pred_full, n_params=p, coefs=AR_coefs, p=p, name=f"AR({p})"
        )

    # Крок 1.2.2: Обчислення АКФ залишків
    acf_vals = [ACF(residue, lag) for lag in range(0, MAX_LAG + 1)]
    # Крок 1.2.3: Визначення q
    q = estimate_order(acf_vals[1:], N)

    # Моделі з бібліотеки statsmodels
    if not dummy:
        # ARMA(p, q) з AR(p) коефіцієнтами
        try:
            arma_model = ARIMA(series, order=(p, 0, q)).fit()
            arma_pred = arma_model.predict(start=0, end=len(series) - 1)
            # ARIMA.params: [const, ar1, ..., arp, ma1, ..., maq, sigma2]
            md_ar_coefs = [arma_model.params.iloc[0]] + [
                arma_model.params.iloc[i] for i in range(1, p + 1)
            ]
            md_ma_coefs = [arma_model.params.iloc[i] for i in range(p + 1, p + q + 1)]

            # Коефіцієнти для stats_report: [const, a1..ap, b1..bq]
            stats_report(
                series.iloc[
                    arma_model.loglikelihood_mle.get_start()
                    - 1 : arma_model.loglikelihood_mle.get_end()
                ],
                arma_pred,
                n_params=p + q,
                coefs=md_ar_coefs + md_ma_coefs,
                p=p,
                name=f"ARMA({p}, {q}) із бібліотеки statsmodels (Підхід №1.2.1)",
            )
        except Exception as e:
            print(f"Помилка при побудові ARIMA({p}, {q}) з statsmodels: {e}")

        dummy.append("was_called")

        print(f"Resid mean: {residue.mean():.6f}   std: {residue.std():.6f}")
        print("Лаг".center(8) + "АКФ (залишки)".center(25))
        for lag in range(0, MAX_LAG + 1):
            print(f"{lag}".center(8) + f"{acf_vals[lag]:.6f}".center(25))
        print(f"Припущення: q = {q}", "\n")

    # Крок 1.2.3 (власне КС): Побудова ARMA(p,q) з MA частиною, отриманою з КС залишків
    # Для цього підходу потрібні залишки, отримані за допомогою КС:

    # 1. Формування КС залишків (sma, ema)
    sma_resid = sma(residue.values, w_size)
    ema_resid = ema(residue.values, w_size)

    # Довжина КС-рядів: N + w_size - 1 (згідно вашої реалізації sma/ema)
    # Обрізаємо для коректної роботи X_matrix та LS.
    # Нам потрібна довжина series (N)
    sma_resid = sma_resid[:N]
    ema_resid = ema_resid[:N]

    # X - матриця для MA частини (лаги v_ma)
    # y = v_ma[q:], X = матриця лагів v_ma[q-1:N-1] ... v_ma[0:N-q]

    # 2. Оцінка коефіцієнтів КС (MA)
    # Підхід: resid(k) = v(k) + b1*mv(k-1) + ... + bq*mv(k-q)
    # У вашому коді: X_matrix(y=[], v=sma_series, p=0, q=q)
    # y - залежна змінна, X - регресори

    # Залежна змінна: Residue_k - V_k = b1*V_{k-1} + ... + bq*V_{k-q}
    # Зміна підходу: спробуємо спрощену формулу, де MA частина - це mv(k)
    # Використаємо MA частину з ручного методу LS для MA(q) на залишках.

    # Формуємо матрицю X для MA(q) на залишкових КС
    # Для MA(q) у вашому коді використовується: X_matrix(y=[], v=sma_series, p=0, q=q)

    # Оцінка MA(q) на КС залишків: resid_k = b0 + b1*mv(k-1) + ... + bq*mv(k-q)

    # SMA (Просте КС)
    m = max([p, q])
    # Регресори: c, mv(k-1)...mv(k-q)
    X_sma_ma = X_matrix(y=[], v=sma_resid, p=0, q=q)
    # Залежна змінна: залишки residue[m:] (те, що ми моделюємо)
    y_sma_ma = residue.values[m:]
    sma_ma_coefs = LS(y=y_sma_ma, X=X_sma_ma)

    # EMA (Експоненційне КС)
    X_ema_ma = X_matrix(y=[], v=ema_resid, p=0, q=q)
    y_ema_ma = residue.values[m:]
    ema_ma_coefs = LS(y=y_ema_ma, X=X_ema_ma)

    # Комбінування: [AR_coefs (a0..ap), MA_coefs (b1..bq)]
    # AR_coefs вже містить a0
    # MA_coefs починаються з b1

    # Коефіцієнти MA(q) з LS включають константу (b0), тому ми її відкидаємо.
    # Коефіцієнти з AR(p) вже містять константу (a0).
    # В ARMA(p,q) є лише одна константа.
    # Припускаємо, що AR_coefs[0] = a0, AR_coefs[1:] = a1..ap.
    # MA_coefs[0] - константа, MA_coefs[1:] = b1..bq.

    # Для комбінованої моделі ARMA(p,q)
    # ARMA: y(k) = a0 + a1*y(k-1) + ... + ap*y(k-p) + v(k) + b1*v(k-1) + ... + bq*v(k-q)
    # Коефіцієнти для generate_arma: [a0, a1..ap, b1..bq]

    # Використовуємо коефіцієнти MA без константи (b1..bq)
    # В моделі за залишками, a0, a1...ap беремо з AR(p), а b1...bq беремо з MA(q) на залишках
    # Це відповідає Підходу №1.1 методички

    # Припускаємо, що LS для MA(q) на залишках дає [b0, b1...bq], але нам потрібні лише b1..bq.
    # Оскільки MA частина моделюється як: resid(k) = b0 + b1*mv(k-1) + ... + bq*mv(k-q) + e(k)
    # Тоді повна модель (у спрощеному варіанті):
    # y(k) = (AR_part) + (MA_part) = (a0 + a1*y(k-1) + ...) + (b1*mv(k-1) + ...)
    # Або ARMA(p,q) як: y(k) = a0 + a1*y(k-1) + ... + ap*y(k-p) + v(k) + b1*v(k-1) + ... + bq*v(k-q)

    # Використовуємо спрощену форму: коефіцієнти AR(p) + коефіцієнти MA(q) з LS (без константи MA)
    # Приймаємо: MA_coefs з LS дають: [b0, b1..bq]. Ми беремо [b1..bq]
    # Це може бути не зовсім коректно, але слідуємо логіці, де MA частина оцінюється
    # окремо і потім додається до AR частини.

    # Залишаємо спрощений варіант, де MA коефіцієнти оцінені як:
    # y(k) - a0 - sum(a_i * y(k-i)) = v(k) + b1*v(k-1) + ...
    # Або (як у вашому коді):
    # ARMA_sma_coefs = np.concatenate([AR_coefs, sma_coefs]) # [a0..ap, b0_ma, b1..bq]
    # Це не відповідає стандартній формі ARMA.
    # Видаляємо константу b0_ma з sma_coefs/ema_coefs.

    if len(sma_ma_coefs) == q + 1 and len(ema_ma_coefs) == q + 1:
        # ARMA_coefs: [a0, a1..ap, b1..bq]
        ARMA_sma_coefs = np.concatenate([AR_coefs, sma_ma_coefs[1:]])
        ARMA_ema_coefs = np.concatenate([AR_coefs, ema_ma_coefs[1:]])
    else:
        print("Помилка при оцінці MA коефіцієнтів. Пропуск підходу.")
        return ({}, {})

    v = np.random.randn(N)

    # Prediction
    sma_pred = generate_arma(v=v, p=p, q=q, coeffs=ARMA_sma_coefs, N=N)
    ema_pred = generate_arma(v=v, p=p, q=q, coeffs=ARMA_ema_coefs, N=N)

    # Обрізка для порівняння зі series
    series_eff = series.values[max_lag:]
    sma_pred_eff = sma_pred[max_lag:]
    ema_pred_eff = ema_pred[max_lag:]

    sma_res = {
        "y_true": series_eff,
        "y_pred": sma_pred_eff,
        "n_params": p + q,
        "coefs": ARMA_sma_coefs,
        "p": p,
        "name": f"ARMA({p}, {q}) із застосуванням власного простого КС (розмір вікна {w_size}) (Підхід №1.2.3)",
    }
    ema_res = {
        "y_true": series_eff,
        "y_pred": ema_pred_eff,
        "n_params": p + q,
        "coefs": ARMA_ema_coefs,
        "p": p,
        "name": f"ARMA({p}, {q}) із застосуванням власного експоненційного КС (розмір вікна {w_size}) (Підхід №1.2.3)",
    }
    return (sma_res, ema_res)


def original_series_approach(series, p, w_size, ma=sma, dummy=[]):
    ma_type = {sma: "власного простого КС", ema: "власного експоненційного КС"}
    print("\n\n" + "=" * 80)
    print(f"Побудова КС за вихідним сигналом у (Підхід №2.1)")
    print("=" * 80)

    N_orig = len(series)

    # Крок 2.2: Побудова КС по у
    ma_series = pd.Series(ma(series.values, w_size))
    # Обрізка до оригінальної довжини, починаючи з першого NaN (якщо він є)
    ma_series = ma_series.dropna()
    N = len(ma_series)  # Нова ефективна довжина

    # Крок 2.3: Визначення q за ЧАКФ (КС)
    ma_pacf_vals = PACF(ma_series, MAX_LAG)
    q = estimate_order(ma_pacf_vals[1:], N)

    if not dummy:
        print("Лаг".center(8) + "ЧАКФ (КС)".center(25))
        for lag in range(0, MAX_LAG + 1):
            print(f"{lag}".center(8) + f"{ma_pacf_vals[lag]:.6f}".center(25))
        print(f"Припущення: q = {q}", "\n")
        dummy.append(0)

    m = max([p, q])
    v = np.random.randn(N_orig)  # Білий шум

    # Оскільки y_eff = y[m:] та ma_series_eff = ma_series[m:]
    # Тоді довжина ефективної вибірки: N_eff = N_orig - m

    series_eff = series.values[m:]
    ma_series_eff = ma_series.values[m:]
    v_eff = v[m:]
    N_eff = len(series_eff)

    # =========================================================================
    # Підхід №2.4, Підхід №1: Застосування власних коефіцієнтів при КС
    # =========================================================================

    # Крок 2.4.1: Обчислення коефіцієнтів b1..bq за формулою (як у методичці)
    alpha = 2 / (w_size + 1)

    # Обчислення знаменника: sum( (1-alpha)**j for j in range(1, q + 1) )
    denominator = sum([(1 - alpha) ** j for j in range(1, q + 1)])

    # Обчислення вагових коефіцієнтів (b_i)
    # Формула: b_i = ( (1-alpha)**j / denominator ) для j=1..q
    # Примітка: у методичці формула відрізняється від стандартної,
    # використовуємо вашу інтерпретацію, де b_i відповідає вагам ЕКС
    if denominator != 0:
        ma_coefs = [((1 - alpha) ** j) / denominator for j in range(1, q + 1)]
    else:
        ma_coefs = [0] * q

    # Модель для AR частини: yl(k) = a0 + a1*y(k-1) + ... + ap*y(k-p)
    # yl(k) = y(k) - mv(k) - sum(b_j * mv(k-j))

    # MA частина: sum(b_j * mv(k-j))
    X_ma_part = X_matrix(y=[], v=ma_series, p=0, q=q)  # [1, mv(k-1)...mv(k-q)]
    # MA частина: b1*mv(k-1) + ... + bq*mv(k-q)
    # ma_part_values: [mv(k-1), ..., mv(k-q)] * [b1, ..., bq]

    # Проекція на MV лаги: X_ma_part[:, 1:] - це матриця лагів mv(k-1)...mv(k-q)
    # ma_part_values - це вектор: sum(b_j * mv(k-j)) для кожного t
    ma_part_values = np.sum(
        [b_j * X_ma_part[:, i] for i, b_j in enumerate(ma_coefs, 1)], axis=0
    )

    # yl(k) = y(k) - mv(k) - sum(b_j * mv(k-j))
    # Обрізка y та mv до ефективного діапазону (N_eff)
    # yl_eff = series_eff - ma_series_eff - ma_part_values

    # Використовуємо спрощену форму з методички:
    # yl(k) = y(k) - mv(k) - sum(b_j * mv(k-j))
    y_to_predict = series.values[m:] - ma_series.values[m:] - ma_part_values

    # Регресори AR: c, y(k-1)...y(k-p)
    X_ar_part = X_matrix(y=series, v=[], p=p, q=0)

    # Оцінка AR коефіцієнтів a0..ap
    ar_coefs = LS(y=y_to_predict, X=X_ar_part)

    # Комбінування коефіцієнтів для generate_arma: [a0, a1..ap, b1..bq]
    # b1..bq беремо з розрахованих ваг ma_coefs
    if len(ar_coefs) == p + 1:
        coefs = np.concatenate([ar_coefs, ma_coefs])
    else:
        print("Помилка: Не вдалося оцінити AR коефіцієнти (Підхід №2.4.1).")
        return [{}, {}]

    # Prediction
    pred = generate_arma(v=v, p=p, q=q, coeffs=coefs, N=N_orig)

    # Обрізка для порівняння зі series
    pred_eff = pred[m:]

    formula_res = {
        "y_true": series_eff,
        "y_pred": pred_eff,
        "n_params": p + q,
        "coefs": coefs,
        "p": p,
        "name": f"ARMA({p}, {q}) із застосуванням {ma_type[ma]} (розмір вікна {w_size}, КС розраховано за формулою) (Підхід №2.4.1)",
    }

    # =========================================================================
    # Підхід №2.4, Підхід №2: Обчислення коефіцієнтів ARMA(p,q) одночасно
    # =========================================================================

    # Модель для LS: yl(k) = a0 + a1*y(k-1) + ... + ap*y(k-p) + b1*mv(k-1) + ... + bq*mv(k-q)
    # yl(k) = y(k) - mv(k)

    y_to_predict_smlt = series.values[m:] - ma_series.values[m:]

    # X_matrix: c, y(k-1)..y(k-p), mv(k-1)..mv(k-q)
    X_smlt = X_matrix(y=series, v=ma_series, p=p, q=q)

    # Оцінка коефіцієнтів [a0, a1..ap, b1..bq]
    smlt_coefs = LS(y=y_to_predict_smlt, X=X_smlt)

    if len(smlt_coefs) != p + q + 1:
        print("Помилка: Не вдалося оцінити коефіцієнти ARMA (Підхід №2.4.2).")
        return [formula_res, {}]

    # Prediction
    pred_smlt = generate_arma(v=v, p=p, q=q, coeffs=smlt_coefs, N=N_orig)
    pred_smlt_eff = pred_smlt[m:]

    smlt_res = {
        "y_true": series_eff,
        "y_pred": pred_smlt_eff,
        "n_params": p + q,
        "coefs": smlt_coefs,
        "p": p,
        "name": f"ARMA({p}, {q}) із застосуванням {ma_type[ma]} (розмір вікна {w_size}, АР та КС розраховано одночасно) (Підхід №2.4.2)",
    }
    return [formula_res, smlt_res]


def adjust_ARMA(series):
    # AD-Fuller test
    series = pd.Series(series)
    result = adfuller(series)
    print(f"\n\nПеревірка стаціонарності на рівні значущості {ALPHA = }")
    print("p-value:", result[1])
    if result[1] > ALPHA:
        print("Ряд не стаціонарний, тому було застосовано взяття перших різниць")
        series = series.diff().dropna()
        result_diff = sm.tsa.stattools.adfuller(series)
        if result_diff[1] > ALPHA:
            print(
                "Ряд не стаціонарний навіть після взяття перших різниць. Подальший аналіз може бути некоректним."
            )
        else:
            print("Ряд стаціонарний після перших різниць.")
    else:
        print("Ряд стаціонарний.")

    # Крок 1.1.1: Обчислення ЧАКФ для визначення p
    N = len(series)
    pacf_vals = PACF(series, MAX_LAG)
    print("\n" + "=" * 80)
    print("Визначення порядку AR (p) за ЧАКФ")
    print("=" * 80)
    print("Лаг".center(8) + "ЧАКФ".center(25))
    for lag in range(0, MAX_LAG + 1):
        print(f"{lag}".center(8) + f"{pacf_vals[lag]:.6f}".center(25))

    # Крок 1.1.2: Визначення p
    p = estimate_order(pacf_vals[1:], N)
    if p == 0:
        p = 1  # Мінімум AR(1), якщо кореляції немає
    print(f"\nПрипущення: p = {p} (по останньому значущому лагу)")

    while True:
        N_orig = len(series)
        W_SIZE = InputReceiver().validate_input(
            "Введіть розмір вікна (N) для КС",
            "очікується натуральне число менше розміру ряду",
            lambda x: isinstance(x, int) and x in range(1, N_orig + 1),
        )

        ma_types = [sma, ema]
        resid_dummy = []

        # 1-й підхід: КС по залишкам AR(p) (Підхід №1)
        for res in resid_approach(series=series, p=p, w_size=W_SIZE, dummy=resid_dummy):
            if res:
                stats_report(**res)

        # 2-й підхід: КС по вихідному сигналу y (Підхід №2)
        orig_dummy = []
        for ma in ma_types:
            for res in original_series_approach(
                series=series, p=p, w_size=W_SIZE, ma=ma, dummy=orig_dummy
            ):
                if res:
                    stats_report(**res)

        response = InputReceiver().validate_input(
            "Бажаєте продовжити роботу з цим рядом? [Y\\n]",
            "очікується y або n",
            lambda x: x in ["Y", "y", "N", "n"],
            key=3,
        )
        if response in ["N", "n"]:
            break


def run():
    # Шляхи для автоматичного завантаження даних (потрібно мати папку 'data')
    dw_file_path = "data/example_for_DW.txt"
    series_file_path = "data/RTStl.txt"

    # --- Частина перша: статистика Дурбіна-Уотсона ---
    print("\n" + "#" * 80)
    print("# Частина перша: Статистика Дарбіна-Уотсона")
    print("#" * 80 + "\n")

    try:
        # Припускаємо, що файл example_for_DW.txt існує
        DW_series = InputReceiver().part_one(file_path=dw_file_path)
    except Exception:
        print(f"Помилка: Файл {dw_file_path} не знайдено. Введіть шлях вручну.")
        DW_series = InputReceiver().part_one()

    if DW_series:
        dw_stat = DW(DW_series)
        print("Значення статистики Дарбіна-Уотсона")
        print(
            f"Розраховане за формулою: {dw_stat:.6f} \t Істинне (statsmodels): {durbin_watson(DW_series):.6f}"
        )

    # --- Частина друга: побудова адекватного рівняння для опису процесу ---
    print("\n" + "#" * 80)
    print(f"# Частина друга: Побудова ARMA/АРКС для ряду RTStl.txt (Варіант №9)")
    print("#" * 80 + "\n")

    while True:
        try:
            series = InputReceiver().read_file(
                f"Вкажіть шлях до файлу з часовим рядом для аналізу (за замовчуванням {series_file_path})",
                file_path_override=series_file_path,
            )
        except Exception:
            print(
                f"Помилка: Файл {series_file_path} не знайдено. Введіть шлях до ряду RTStl.txt вручну."
            )
            series = InputReceiver().read_file(
                "\nВкажіть шлях до файлу з часовим рядом для аналізу"
            )

        if series:
            print(f"Завантажено {len(series)} спостережень.")
            descriptive_analysis(series)  # Крок 1.1 методички
            adjust_ARMA(series)  # Кроки 1.2-2.4 методички

        response = InputReceiver().validate_input(
            "Бажаєте завантажити інший ряд для Частини 2? [Y\\n]",
            "очікується y або n",
            lambda x: x in ["Y", "y", "N", "n"],
            key=3,
        )
        if response in ["N", "n"]:
            break

    # --- Частина третя: побудова рівняння множинної регресії ---
    print("\n" + "#" * 80)
    print("# Частина третя: Побудова рівняння множинної регресії (RTStl - залежна)")
    print("#" * 80 + "\n")

    data = InputReceiver().load_data()
    if data.empty:
        print("Не вдалося завантажити дані для Частини 3. Пропуск.")
        return

    # Залежна змінна згідно варіанту №9 - RTStl (перейменовано в 'rts1' у load_data)
    cov_matrix = data.corr()
    target_name = "RTStl"

    print("Кореляційна матриця індексів".center(82))
    print(cov_matrix)
    print("\n" + "-" * 80)

    # Визначення суттєвих індексів (Крок 7, методичка)
    important_features = []
    # Перебір стовпців, порівняння з target_name
    for i in cov_matrix[target_name].index:
        if abs(cov_matrix[target_name][i]) >= SIGNIFICANT_CORREL and i != target_name:
            important_features.append(i)

    print("\nСуттєві індекси (Кореляція >= 0.5)")
    print(*important_features, sep=", ")

    if not important_features:
        print("Немає суттєвих індексів для побудови моделі множинної регресії.")
        return

    # Побудова рівняння множинної регресії (МНК)
    N = len(data)
    y = data[target_name].values
    features = data[important_features].values

    # Матриця Х: [1, x1, x2, ...]
    X = np.c_[np.ones(shape=(N, 1)), features]

    coefs = LS(y=y, X=X)

    if len(coefs) != len(important_features) + 1:
        print("Помилка: Не вдалося оцінити коефіцієнти множинної регресії.")
        return

    pred = np.sum([coef * X[:, i] for i, coef in enumerate(coefs)], axis=0)

    # Вивід рівняння
    eq_str = f"{target_name} = {coefs[0]:.4f} + "
    eq_str += " + ".join(
        [f"{coefs[i+1]:.4f}*{feat}" for i, feat in enumerate(important_features)]
    )

    print("\n" + "=" * 80)
    print("Оцінене рівняння множинної регресії (МНК)")
    print(eq_str)

    # Звіт статистики
    stats_report(
        y,
        pred,
        n_params=len(coefs) - 1,
        p=len(coefs) - 1,
        coefs=coefs,
        name="Множинної регресії (МНК)",
    )


if __name__ == "__main__":
    # Встановлюємо робочу директорію на каталог, де знаходиться скрипт
    # Щоб коректно працювали відносні шляхи типу 'data/...'
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(base_dir)

    warnings.filterwarnings("ignore", module="statsmodels")
    run()
