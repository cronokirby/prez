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
    data = PollData.from_csv("./polls.csv")
    observed = [data.count_D, data.count_R, data.count_O]
    model = pm.Model()
    with model:
        probs = pm.Dirichlet("probs", [1.0, 1.0, 1.0])
        results = pm.Multinomial("results", data.size, probs, observed=observed)
        trace = pm.sample(SAMPLE_SIZE)
    pm.traceplot(trace)
    plt.gcf().savefig(".out/posterior.png")
