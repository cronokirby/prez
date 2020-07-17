from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import csv


def percent_from_str(s):
    return float(s.strip("%")) / 100


@dataclass
class PollData:
    """
    Represents aggregate poll data.

    size: the total number of people polled
    count_D: the total number of Democratic choices
    count_R: the total number of Republican choices
    count_O: the total number of choices for another candidate / undecided
    """

    size: int
    count_D: int
    count_R: int
    count_O: int

    @staticmethod
    def from_csv(path):
        with open(path, "r") as fp:
            reader = csv.reader(fp, delimiter=",")
            total = 0
            total_D = 0
            total_R = 0
            for row in reader:
                # We only want the national counts, for now
                if row[0] != "National":
                    continue
                poll_count = int(row[1])
                percent_D = percent_from_str(row[3])
                percent_R = percent_from_str(row[4])
                total += poll_count
                total_D += poll_count * percent_D
                total_R += poll_count * percent_R
            total_D = int(total_D)
            total_R = int(total_R)
            return PollData(total, total_D, total_R, total - total_D - total_R)


SAMPLE_SIZE = 1000

if __name__ == "__main__":
    model = pm.Model()
    with model:
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        mu_b = pm.Normal("mu_b", mu=0, sigma=1, shape=(4,))
        state_probs = pm.Deterministic("state_probs", pm.math.sigmoid(mu_a + mu_b))
        texas_results = pm.Binomial("tx_results", 100, state_probs[0], observed=40)
        p_likes_biden = pm.Deterministic(
            "p_likes_biden", pm.math.dot([0.7, 0.3, 0.2, 0.1], state_probs)
        )
        biden_state_wins = pm.Bernoulli(
            "biden_state_wins", p=state_probs.flatten(), shape=(4,)
        )
        biden_electors = pm.Deterministic(
            "biden_electors", pm.math.dot([4, 0, 0, 0], biden_state_wins)
        )
        results = pm.Binomial("results", 100, p_likes_biden, observed=60)
        trace = pm.sample()
    pm.plots.plot_posterior(trace, var_names="biden_electors")
    plt.gcf().savefig(".out/electors.png")
    pm.plots.forestplot(
        trace,
        var_names="state_probs",
        combined=True,
        colors=["red", "blue", "blue", "blue"],
    )
    plt.gcf().savefig(".out/posterior.png")
