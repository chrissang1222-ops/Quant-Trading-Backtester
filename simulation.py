import numpy as np

def gbm(S0=100, mu=0.05, sigma=0.2, T=1, steps=252):
    dt = T / steps
    prices = [S0]

    for _ in range(steps):
        shock = np.random.normal(0, 1)
        S_next = prices[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)
        prices.append(S_next)

    return prices