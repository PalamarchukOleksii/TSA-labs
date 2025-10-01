import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

true_params = {
    'a0': 0,
    'a1': 0.22,
    'a2': -0.18,
    'a3': 0.08,
    'b1': 0.5,
    'b2': 0.25,
    'b3': 0.25
}

def load_data(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не знайдено: {filepath}")
    return np.loadtxt(filepath)

def least_squares(y, v, p, q):
    max_lag = max(p, q)
    n = len(y)
    X = []
    Y = []
    for k in range(max_lag, n):
        row = [1]
        row += [y[k-i] for i in range(1, p+1)]
        row += [v[k-i] for i in range(1, q+1)]
        X.append(row)
        Y.append(y[k])
    X = np.array(X)
    Y = np.array(Y)
    theta = np.linalg.solve(X.T @ X, X.T @ Y)
    return theta

def recursive_least_squares(y, v, p, q, lambda_=0.98):
    max_lag = max(p, q)
    n = len(y)
    num_params = 1 + p + q
    theta = np.zeros(num_params)
    P = np.eye(num_params) * 1000
    theta_history = []
    for k in range(max_lag, n):
        phi = np.array([1] + [y[k-i] for i in range(1, p+1)] + [v[k-i] for i in range(1, q+1)])
        r = P @ phi
        s = phi.T @ r
        gamma = r / (lambda_ + s)
        error = y[k] - phi @ theta
        theta = theta + gamma * error
        P = (P - np.outer(gamma, r)) / lambda_
        theta_history.append(theta.copy())
    return theta, np.array(theta_history)

def compute_metrics(y, v, theta, p, q):
    max_lag = max(p, q)
    y_pred = []
    errors = []
    for k in range(max_lag, len(y)):
        pred = theta[0]
        pred += sum(theta[i] * y[k-i] for i in range(1, p+1))
        pred += sum(theta[p+i] * v[k-i] for i in range(1, q+1))
        y_pred.append(pred)
        errors.append(y[k] - pred)
    y_actual = y[max_lag:]
    y_pred = np.array(y_pred)
    errors = np.array(errors)
    SSE = np.sum(errors**2)
    SST = np.sum((y_actual - np.mean(y_actual))**2)
    R2 = 1 - SSE/SST if SST > 0 else 0
    n_obs = len(y_actual)
    num_params = 1 + p + q
    AIC = n_obs * np.log(SSE/n_obs) + 2*num_params
    return SSE, R2, AIC

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    print("Завантаження даних...")
    try:
        y = load_data('data/y.txt')
        v = load_data('data/v.txt')
        print(f"Завантажено {len(y)} спостережень")
    except FileNotFoundError:
        print("Файли y.txt або v.txt не знайдено! Генеруємо тестові дані...")
        n = 100
        u = np.random.randn(n)*0.5
        v = np.zeros(n)
        y = np.zeros(n)
        for k in range(n):
            v[k] = u[k]
            if k >= 1: v[k] += true_params['b1'] * u[k-1]
            if k >= 2: v[k] += true_params['b2'] * u[k-2]
            if k >= 3: v[k] += true_params['b3'] * u[k-3]
        for k in range(n):
            y[k] = true_params['a0'] + v[k]
            if k >= 1: y[k] += true_params['a1'] * y[k-1]
            if k >= 2: y[k] += true_params['a2'] * y[k-2]
            if k >= 3: y[k] += true_params['a3'] * y[k-3]
        np.savetxt(os.path.join(script_dir, 'data/y.txt'), y)
        np.savetxt(os.path.join(script_dir, 'data/v.txt'), v)
        print("Тестові дані збережено у data/y.txt та data/v.txt")

    print("\nСправжні параметри моделі (Команда 1):")
    for key, val in true_params.items():
        print(f"{key} = {val}")

    results = []
    for p in range(1,4):
        for q in range(1,4):
            model_name = f"АРКС({p},{q})"
            print(f"\nОцінювання {model_name}...")

            theta_ls = least_squares(y, v, p, q)
            sse_ls, r2_ls, aic_ls = compute_metrics(y, v, theta_ls, p, q)

            theta_rls, theta_history = recursive_least_squares(y, v, p, q)
            sse_rls, r2_rls, aic_rls = compute_metrics(y, v, theta_rls, p, q)

            results.append({
                'Модель': model_name,
                'p': p,
                'q': q,
                'МНК_SSE': sse_ls,
                'МНК_R2': r2_ls,
                'МНК_AIC': aic_ls,
                'РМНК_SSE': sse_rls,
                'РМНК_R2': r2_rls,
                'РМНК_AIC': aic_rls,
                'theta_ls': theta_ls,
                'theta_rls': theta_rls,
                'theta_history': theta_history
            })

    df = pd.DataFrame(results)

    print("\nРЕЗУЛЬТАТИ ОЦІНЮВАННЯ")
    print(df[['Модель','МНК_SSE','МНК_R2','МНК_AIC','РМНК_SSE','РМНК_R2','РМНК_AIC']])

    best_ls = df.loc[df['МНК_AIC'].idxmin()]
    best_rls = df.loc[df['РМНК_AIC'].idxmin()]

    print("\nНАЙКРАЩІ МОДЕЛІ (за AIC):")
    print(f"МНК: {best_ls['Модель']} (AIC={best_ls['МНК_AIC']:.2f})")
    print(f"РМНК: {best_rls['Модель']} (AIC={best_rls['РМНК_AIC']:.2f})")

    fig, axes = plt.subplots(3,3,figsize=(15,12))
    fig.suptitle('Збіжність параметрів РМНК для різних моделей', fontsize=16)
    for idx, result in enumerate(results):
        ax = axes[idx // 3, idx % 3]
        history = result['theta_history']
        for i in range(history.shape[1]):
            ax.plot(history[:,i], label=f'θ{i}', alpha=0.7)
        ax.set_title(result['Модель'])
        ax.set_xlabel('Крок k')
        ax.set_ylabel('Значення параметра')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,'convergence_plots.png'), dpi=150)
    print(f"Збережено: {os.path.join(result_dir,'convergence_plots.png')}")

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    models = df['Модель'].values
    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x-width/2, df['МНК_SSE'], width, label='МНК', alpha=0.8)
    axes[0].bar(x+width/2, df['РМНК_SSE'], width, label='РМНК', alpha=0.8)
    axes[0].set_title('SSE')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x-width/2, df['МНК_R2'], width, label='МНК', alpha=0.8)
    axes[1].bar(x+width/2, df['РМНК_R2'], width, label='РМНК', alpha=0.8)
    axes[1].set_title('R²')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(x-width/2, df['МНК_AIC'], width, label='МНК', alpha=0.8)
    axes[2].bar(x+width/2, df['РМНК_AIC'], width, label='РМНК', alpha=0.8)
    axes[2].set_title('AIC')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,'metrics_comparison.png'), dpi=150)
    print(f"Збережено: {os.path.join(result_dir,'metrics_comparison.png')}")

    df[['Модель','МНК_SSE','МНК_R2','МНК_AIC','РМНК_SSE','РМНК_R2','РМНК_AIC']].to_csv(os.path.join(result_dir,'results.csv'), index=False)
    print(f"Збережено: {os.path.join(result_dir,'results.csv')}")

    plt.show()

if __name__ == "__main__":
    main()
