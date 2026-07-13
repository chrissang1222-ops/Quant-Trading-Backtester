import numpy as np


def gbm(
    S0=100,
    mu=0.05,
    sigma=0.2,
    T=1,
    steps=252,
    simulations=1000
):

    dt = T / steps

    paths = np.zeros((simulations, steps + 1))

    paths[:, 0] = S0

    for i in range(simulations):
        for t in range(1, steps + 1):

            shock = np.random.normal(0, 1)

            paths[i, t] = (
                paths[i, t-1]
                * np.exp(
                    (mu - 0.5 * sigma**2) * dt
                    + sigma * np.sqrt(dt) * shock
                )
            )

    return paths