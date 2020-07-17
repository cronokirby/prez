import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pymc3 as pm
import pandas as pd
import math


# The variance for the logits of probabilities
VARIANCE = 0.05
# The correlation between logits of different states
STATE_CORR = 0.5
# The number of samples to make for the final empirical odds
SAMPLE_SIZE = 1000
# The turnout we expect, also acts as a kind of normalizer. The higher the turnout
# the less likely to win a state where you're behind in opinion
TURNOUT = 100
# The number of electors you need to win
ELECTORS_TO_WIN = 270


def percent_from_str(s):
    return float(s.strip("%")) / 100


def read_state_info(csv_file):
    df = pd.read_csv(csv_file)
    df.D_2016 = df.D_2016.apply(percent_from_str)
    df.R_2016 = df.R_2016.apply(percent_from_str)
    df["D_Prob"] = df.D_2016 / (df.D_2016 + df.R_2016)
    df["Weight"] = df.Adults / df.Adults.sum()
    # Making sure we have a consistent order later
    return df.sort_values(by=["State"])


def get_state_probs(df):
    return np.array(df[df.State != "Popular"].D_Prob)


def get_state_weights(df):
    return np.array(df[df.State != "Popular"].Weight)


def get_state_electors(df):
    return np.array(df[df.State != "Popular"].Electors)


def get_popular_percentage(df):
    return df[df.State == "Popular"].D_Prob[0]


def read_poll_info(csv_file):
    df = pd.read_csv(csv_file)
    df.D = df.D.apply(percent_from_str)
    df.R = df.R.apply(percent_from_str)
    df["Effective_Size"] = df.Size * (df.D + df.R)
    df["D_Count"] = df.Size * df.D
    return (df.Effective_Size.sum(), df.D_Count.sum())


def logit(p):
    return np.log(p) - np.log(1 - p)


def bin_conf_95(p_hat, n_samples):
    return 1.96 * math.sqrt(p_hat * (1 - p_hat) / n_samples)


def simulate(ps, state_electors):
    wins = np.random.binomial(TURNOUT, ps) > TURNOUT / 2
    electors = np.dot(wins, state_electors)
    p_hat = np.count_nonzero(electors >= ELECTORS_TO_WIN) / len(electors)
    return (wins, electors, p_hat, bin_conf_95(p_hat, len(electors)))


if __name__ == "__main__":
    state_df = read_state_info("state_info.csv")
    state_votes = get_state_probs(state_df)
    state_weights = get_state_weights(state_df)
    state_electors = get_state_electors(state_df)
    # If this isn't 50 something is wrong
    num_states = len(state_votes)
    popular_p = get_popular_percentage(state_df)
    polled_size, polled_d = read_poll_info("polls.csv")

    model = pm.Model()
    with model:
        mu_a = pm.Normal("mu_a", mu=logit(popular_p), sigma=pm.math.sqrt(VARIANCE))
        # Gives us a symmetric matrix with the right variances and covariances
        cov = VARIANCE * np.eye(num_states) + VARIANCE * STATE_CORR * (
            1 - np.eye(num_states)
        )
        mu_b = pm.MvNormal("mu_b", mu=logit(state_votes), cov=cov, shape=(num_states,))
        state_probs = pm.Deterministic("state_probs", pm.math.sigmoid(mu_a + mu_b))
        p_likes_biden = pm.Deterministic(
            "p_likes_biden", pm.math.dot(state_weights, state_probs)
        )
        polled = pm.Binomial("polled", polled_size, p_likes_biden, observed=polled_d)
        trace = pm.sample()
    pm.traceplot(trace, combined=True)
    plt.gcf().savefig(".out/posterior.png")
    pm.plots.forestplot(
        trace,
        var_names="state_probs",
        combined=True,
        colors=["red", "blue", "blue", "blue"],
    )
    plt.gcf().savefig(".out/state_probabilities.png")
    state_probs = np.array(
        pm.sample_posterior_predictive(
            trace, var_names=["state_probs"], samples=SAMPLE_SIZE, model=model
        )["state_probs"]
    )
    wins, electors, p_hat, p_delta = simulate(state_probs, state_electors)
    state_ps = np.count_nonzero(wins, axis=0) / SAMPLE_SIZE
    state_probs = sorted(
        list(zip(list(state_df[state_df.State != "Popular"].State), list(state_ps))),
        key=lambda x: x[1],
    )
    plt.clf()
    fig = plt.subplots(1, 1)
    plt.hist(electors, bins=100)
    plt.gcf().savefig(".out/electors.png")
    print(f"\nBiden wins with {p_hat * 100:.2f} ± {p_delta * 100:.2f}%")
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    rtb = mcolors.LinearSegmentedColormap.from_list("", [(0, "red"), (1, "blue")])
    ax.barh(
        list(np.arange(num_states)),
        [x[1] for x in state_probs],
        align="center",
        color=rtb(np.array([x[1] for x in state_probs])),
    )
    ax.set_yticks(list(np.arange(num_states)))
    ax.set_yticklabels([x[0] for x in state_probs])
    fig.savefig(".out/states.png")

