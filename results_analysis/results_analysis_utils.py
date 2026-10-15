from matplotlib import pyplot as plt
from configs.config import *
import pandas as pd
'''
Goal: Take the simulated result paths and compute statistics
'''

def mean_and_variance(sim_paths, plot=True):
    means = []
    variances = []
    q_5s = []
    q_95s = []

    for idx, t in enumerate(time_grid):
        allocs_at_t = sim_paths[:, idx]
        portfolio_value = np.sum(allocs_at_t, 1)
        mean = np.mean(portfolio_value)
        variance = np.var(portfolio_value)
        q_5, q_95 = np.quantile(portfolio_value, [0.5, 0.95])

        means.append(mean)
        variances.append(variance)
        q_5s.append(q_5)
        q_95s.append(q_95)

    df = pd.DataFrame({'Mean': means, 'Variance': variances, '5%-quantile': q_5s, '95%-quantile': q_95s})

    if plot:
        ax = df['Mean'].plot(label='Mean', figsize=(8, 4))
        ax.fill_between(df.index, df['5%-quantile'], df['95%-quantile'], alpha=0.3, label='5%–95%')
        ax.set(xlabel='Time step', ylabel='Portfolio value')
        ax.legend()
        plt.show()

    return df



