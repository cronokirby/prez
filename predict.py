import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import math


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


def logit(p):
    return np.log(p) - np.log(1 - p)


# The variance for the logits of probabilities
VARIANCE = 0.05
# The correlation between logits of different states
STATE_CORR = 0.5
# The number of samples to make for the final empirical odds
SAMPLE_SIZE = 10000

if __name__ == "__main__":
    state_df = read_state_info("state_info.csv")
    state_votes = get_state_probs(state_df)
    state_weights = get_state_weights(state_df)
    state_electors = get_state_electors(state_df)
    # If this isn't 50 something is wrong
    num_states = len(state_votes)
    popular_p = get_popular_percentage(state_df)

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
        biden_state_wins = pm.Bernoulli(
            "biden_state_wins", p=state_probs.flatten(), shape=(num_states,)
        )
        biden_electors = pm.Deterministic(
            "biden_electors", pm.math.dot(state_electors, biden_state_wins)
        )
        trace = pm.sample()
    pm.plots.plot_posterior(trace, var_names="biden_electors")
    plt.gcf().savefig(".out/electors.png")
    pm.plots.forestplot(
        trace,
        var_names="state_probs",
        combined=True,
        colors=["red", "blue", "blue", "blue"],
    )
    plt.gcf().savefig(".out/state_probabilities.png")
    biden_electors = np.array(
        pm.sample_posterior_predictive(
            trace, var_names=["biden_electors"], samples=SAMPLE_SIZE, model=model
        )["biden_electors"]
    )
    biden_wins = biden_electors >= 270
    p_biden_wins = np.count_nonzero(biden_wins) / SAMPLE_SIZE
    # 95% confidence interval
    confidence = 1.96 * math.sqrt(p_biden_wins * (1 - p_biden_wins) / SAMPLE_SIZE)
    print(f"\nProbability of Biden Winning: {p_biden_wins * 100:.2f}% ± {confidence * 100:.2f}%")
