import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

P_BIDEN = 0.5
P_TRUMP = 0.4

if __name__ == '__main__':
    samples = np.random.multinomial(1000, [P_BIDEN, P_TRUMP, 1.0 - P_BIDEN - P_TRUMP])
    print(samples)
    model = pm.Model()
    with model:
        probs = pm.Dirichlet('probs', [1.0, 1.0, 1.0])
        results = pm.Multinomial('results', 1000, probs, observed=samples)
        trace = pm.sample(1000)
    pm.traceplot(trace)
    plt.gcf().savefig('.out/foo.png')
