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

def load_file(filename):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return np.loadtxt(filepath)

def load_data():
    try:
        y = load_file('data/y.txt')
        v = load_file('data/v.txt')
        print(f"Loaded {len(y)} observations from files.")
    except FileNotFoundError:
        print("Files not found. Generating synthetic data...")
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
        print(f"Generated {n} synthetic observations.")
    return y, v

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
        theta += gamma * error
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
    y, v = load_data()
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(result_dir, exist_ok=True)
    results = []
    for p in range(1,4):
        for q in range(1,4):
            model = f"ARMA({p},{q})"
            theta_ls = least_squares(y, v, p, q)
            sse_ls, r2_ls, aic_ls = compute_metrics(y, v, theta_ls, p, q)
            theta_rls, theta_hist = recursive_least_squares(y, v, p, q)
            sse_rls, r2_rls, aic_rls = compute_metrics(y, v, theta_rls, p, q)
            results.append({
                'Model': model, 'p': p, 'q': q,
                'LS_SSE': sse_ls, 'LS_R2': r2_ls, 'LS_AIC': aic_ls,
                'RLS_SSE': sse_rls, 'RLS_R2': r2_rls, 'RLS_AIC': aic_rls,
                'theta_ls': theta_ls, 'theta_rls': theta_rls, 'theta_history': theta_hist
            })
    df = pd.DataFrame(results)
    print("\nEvaluation Metrics:")
    print(df[['Model','LS_SSE','LS_R2','LS_AIC','RLS_SSE','RLS_R2','RLS_AIC']])
    best_ls = df.loc[df['LS_AIC'].idxmin()]
    best_rls = df.loc[df['RLS_AIC'].idxmin()]
    print(f"\nBest Models by AIC:\nLS: {best_ls['Model']} (AIC={best_ls['LS_AIC']:.2f})\nRLS: {best_rls['Model']} (AIC={best_rls['RLS_AIC']:.2f})")

    p, q = 3, 3
    model_name = f"ARX({p},{q})"

    theta_ls = least_squares(y, v, p, q)
    theta_rls, theta_history = recursive_least_squares(y, v, p, q)

    param_names = ['a0'] + [f'a{i}' for i in range(1, p + 1)] + [f'b{j}' for j in range(1, q + 1)]
    true_theta = [true_params[name] for name in param_names]
    k_range = np.arange(theta_history.shape[0]) + max(p, q)

    coeff_dir = os.path.join(result_dir, 'coeff_plots_ARX_3_3')
    os.makedirs(coeff_dir, exist_ok=True)

    for i, name in enumerate(param_names):
        plt.figure(figsize=(10, 3))

        plt.plot(k_range, theta_history[:, i], color='green', label='RLS')
        plt.hlines(theta_ls[i], k_range[0], k_range[-1], color='red', linestyle='--', label='LS')
        plt.hlines(true_theta[i], k_range[0], k_range[-1], color='black', linestyle='-', label='True')

        plt.xlabel("Discrete time $k$")
        plt.ylabel("Value")
        plt.title(f"Coefficient {name}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = os.path.join(coeff_dir, f"coef_{name}_ARX_3_3.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close()

    models = df['Model'].values

    metrics_dir = os.path.join(result_dir, 'metrics_plots')
    os.makedirs(metrics_dir, exist_ok=True)

    x = np.arange(len(models))

    plt.figure(figsize=(8, 5))
    plt.plot(x, df['RLS_SSE'], 'go-', label='S obtained by RLS')
    plt.plot(x, df['LS_SSE'], 'r^--', label='S obtained by LS')
    plt.xticks(x, models, rotation=45)
    plt.ylabel("S (sum of squared errors)")
    plt.title("Change of sum of squared errors for ARX(p,q) models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'sse_vs_model.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, df['RLS_R2'], 'go-', label='R² obtained by RLS')
    plt.plot(x, df['LS_R2'], 'r^--', label='R² obtained by LS')
    plt.xticks(x, models, rotation=45)
    plt.ylabel("Coefficient of determination R²")
    plt.title("Change of R² for ARX(p,q) models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'r2_vs_model.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, df['RLS_AIC'], 'go-', label='AIC obtained by RLS')
    plt.plot(x, df['LS_AIC'], 'r^--', label='AIC obtained by LS')
    plt.xticks(x, models, rotation=45)
    plt.ylabel("Akaike Information Criterion (AIC)")
    plt.title("Change of AIC for ARX(p,q) models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'aic_vs_model.png'), dpi=150)
    plt.close()

    df[['Model','LS_SSE','LS_R2','LS_AIC','RLS_SSE','RLS_R2','RLS_AIC']].to_csv(os.path.join(result_dir,'results.csv'), index=False)

if __name__ == "__main__":
    main()
