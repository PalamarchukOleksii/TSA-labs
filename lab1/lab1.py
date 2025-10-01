import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Справжні параметри для команди №1
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
    """
    Завантаження даних з текстового файлу.
    Шлях будується відносно місця знаходження цього скрипта.
    """
    # Отримати папку, де лежить скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Формуємо повний шлях до файлу
    filepath = os.path.join(script_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не знайдено: {filepath}")
    return np.loadtxt(filepath)

def least_squares(y, v, p, q):
    """Метод найменших квадратів (МНК)"""
    max_lag = max(p, q)
    n = len(y)
    num_params = 1 + p + q
    
    # Формування матриці X та вектора Y
    X = []
    Y = []
    
    for k in range(max_lag, n):
        row = [1]  # a0
        for i in range(1, p + 1):
            row.append(y[k - i])
        for i in range(1, q + 1):
            row.append(v[k - i])
        X.append(row)
        Y.append(y[k])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Обчислення theta = (X^T * X)^(-1) * X^T * Y
    theta = np.linalg.solve(X.T @ X, X.T @ Y)
    
    return theta

def recursive_least_squares(y, v, p, q, lambda_=0.98):
    """Рекурсивний метод найменших квадратів (РМНК)"""
    max_lag = max(p, q)
    n = len(y)
    num_params = 1 + p + q
    
    # Ініціалізація
    theta = np.zeros(num_params)
    P = np.eye(num_params) * 1000
    
    theta_history = []
    
    for k in range(max_lag, n):
        # Формування вектора phi
        phi = np.array([1] + 
                       [y[k - i] for i in range(1, p + 1)] + 
                       [v[k - i] for i in range(1, q + 1)])
        
        # Обчислення r = P * phi
        r = P @ phi
        
        # Обчислення s = phi^T * P * phi
        s = phi.T @ r
        
        # Обчислення gamma = r / (lambda + s)
        gamma = r / (lambda_ + s)
        
        # Обчислення похибки
        error = y[k] - phi @ theta
        
        # Оновлення theta
        theta = theta + gamma * error
        
        # Оновлення P
        P = (P - np.outer(gamma, r)) / lambda_
        
        theta_history.append(theta.copy())
    
    return theta, np.array(theta_history)

def compute_metrics(y, v, theta, p, q):
    """Обчислення метрик якості моделі"""
    max_lag = max(p, q)
    n = len(y)
    
    y_pred = []
    errors = []
    
    for k in range(max_lag, n):
        # Прогноз
        pred = theta[0]
        for i in range(1, p + 1):
            pred += theta[i] * y[k - i]
        for i in range(1, q + 1):
            pred += theta[p + i] * v[k - i]
        
        y_pred.append(pred)
        errors.append(y[k] - pred)
    
    y_actual = y[max_lag:]
    y_pred = np.array(y_pred)
    errors = np.array(errors)
    
    # Метрики
    SSE = np.sum(errors ** 2)
    SST = np.sum((y_actual - np.mean(y_actual)) ** 2)
    R2 = 1 - SSE / SST if SST > 0 else 0
    
    num_params = 1 + p + q
    n_obs = len(y_actual)
    AIC = n_obs * np.log(SSE / n_obs) + 2 * num_params
    
    return SSE, R2, AIC

def main():
    # Завантаження даних
    print("Завантаження даних...")
    try:
        y = load_data('data/y.txt')
        v = load_data('data/v.txt')
        print(f"Завантажено {len(y)} спостережень")
    except FileNotFoundError:
        print("Файли y.txt або v.txt не знайдено!")
        print("Генеруємо тестові дані...")
        
        # Генерація тестових даних
        n = 100
        u = np.random.randn(n) * 0.5
        v = np.zeros(n)
        y = np.zeros(n)
        
        # Генерація v
        for k in range(n):
            v[k] = u[k]
            if k >= 1: v[k] += true_params['b1'] * u[k-1]
            if k >= 2: v[k] += true_params['b2'] * u[k-2]
            if k >= 3: v[k] += true_params['b3'] * u[k-3]
        
        # Генерація y
        for k in range(n):
            y[k] = true_params['a0'] + v[k]
            if k >= 1: y[k] += true_params['a1'] * y[k-1]
            if k >= 2: y[k] += true_params['a2'] * y[k-2]
            if k >= 3: y[k] += true_params['a3'] * y[k-3]
        
        # Збереження тестових даних
        np.savetxt('y.txt', y)
        np.savetxt('v.txt', v)
        print("Тестові дані збережено у y.txt та v.txt")
    
    # Вивід справжніх параметрів
    print("\n" + "="*50)
    print("Справжні параметри моделі (Команда 1):")
    print("="*50)
    for key, value in true_params.items():
        print(f"{key} = {value}")
    
    # Оцінювання моделей
    results = []
    
    print("\n" + "="*50)
    print("Оцінювання моделей...")
    print("="*50)
    
    for p in range(1, 4):
        for q in range(1, 4):
            model_name = f"АРКС({p},{q})"
            print(f"\nОцінювання {model_name}...")
            
            # МНК
            theta_ls = least_squares(y, v, p, q)
            sse_ls, r2_ls, aic_ls = compute_metrics(y, v, theta_ls, p, q)
            
            # РМНК
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
    
    # Створення таблиці результатів
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТИ ОЦІНЮВАННЯ")
    print("="*80)
    print(df[['Модель', 'МНК_SSE', 'МНК_R2', 'МНК_AIC', 
              'РМНК_SSE', 'РМНК_R2', 'РМНК_AIC']].to_string(index=False))
    
    # Визначення найкращої моделі
    best_ls = df.loc[df['МНК_AIC'].idxmin()]
    best_rls = df.loc[df['РМНК_AIC'].idxmin()]
    
    print("\n" + "="*50)
    print("НАЙКРАЩІ МОДЕЛІ (за критерієм AIC)")
    print("="*50)
    print(f"МНК:  {best_ls['Модель']} (AIC = {best_ls['МНК_AIC']:.2f})")
    print(f"РМНК: {best_rls['Модель']} (AIC = {best_rls['РМНК_AIC']:.2f})")
    
    # Вивід оцінених параметрів найкращої моделі
    print("\n" + "="*50)
    print(f"ОЦІНЕНІ ПАРАМЕТРИ для {best_rls['Модель']}")
    print("="*50)
    print("\nМНК:")
    for i, val in enumerate(best_ls['theta_ls']):
        print(f"  θ{i} = {val:.4f}")
    
    print("\nРМНК:")
    for i, val in enumerate(best_rls['theta_rls']):
        print(f"  θ{i} = {val:.4f}")
    
    # Побудова графіків
    print("\nПобудова графіків...")
    
    # Графік 1: Збіжність параметрів РМНК
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Збіжність параметрів РМНК для різних моделей', fontsize=16)
    
    for idx, result in enumerate(results):
        ax = axes[idx // 3, idx % 3]
        history = result['theta_history']
        
        for i in range(history.shape[1]):
            ax.plot(history[:, i], label=f'θ{i}', alpha=0.7)
        
        ax.set_title(result['Модель'])
        ax.set_xlabel('Крок k')
        ax.set_ylabel('Значення параметра')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_plots.png', dpi=150)
    print("Збережено: convergence_plots.png")
    
    # Графік 2: Порівняння метрик
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = df['Модель'].values
    x = np.arange(len(models))
    width = 0.35
    
    # SSE
    axes[0].bar(x - width/2, df['МНК_SSE'], width, label='МНК', alpha=0.8)
    axes[0].bar(x + width/2, df['РМНК_SSE'], width, label='РМНК', alpha=0.8)
    axes[0].set_xlabel('Модель')
    axes[0].set_ylabel('SSE')
    axes[0].set_title('Сума квадратів похибок')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R²
    axes[1].bar(x - width/2, df['МНК_R2'], width, label='МНК', alpha=0.8)
    axes[1].bar(x + width/2, df['РМНК_R2'], width, label='РМНК', alpha=0.8)
    axes[1].set_xlabel('Модель')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Коефіцієнт детермінації')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AIC
    axes[2].bar(x - width/2, df['МНК_AIC'], width, label='МНК', alpha=0.8)
    axes[2].bar(x + width/2, df['РМНК_AIC'], width, label='РМНК', alpha=0.8)
    axes[2].set_xlabel('Модель')
    axes[2].set_ylabel('AIC')
    axes[2].set_title('Критерій Акайке')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150)
    print("Збережено: metrics_comparison.png")
    
    # Збереження результатів у CSV
    df[['Модель', 'МНК_SSE', 'МНК_R2', 'МНК_AIC', 
        'РМНК_SSE', 'РМНК_R2', 'РМНК_AIC']].to_csv('results.csv', index=False)
    print("Збережено: results.csv")
    
    print("\n" + "="*50)
    print("ВИСНОВКИ")
    print("="*50)
    print("1. РМНК дає кращі результати для онлайн-оцінювання")
    print("2. МНК потребує всіх даних одночасно")
    print("3. Використання кількох критеріїв запобігає перенавчанню")
    print("4. AIC враховує складність моделі")
    
    plt.show()

if __name__ == "__main__":
    main()
