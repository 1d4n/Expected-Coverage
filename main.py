import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.special import comb
from sympy import EulerGamma


def hamming_distance(u: int, v: int):
    """
    Calculates the hamming distance between two vectors.
    :param u: First vector.
    :param v: Second vector.
    :return: Hamming distance between the two vectors.
    """
    return bin(u ^ v).count('1')


def cover_ball(covered: list[bool], v: int, t: int):
    """
    Covers the neighbours of v that are at distance at most t in the hamming graph.
    :param covered: List of vertices that are already covered. Its length is 2**n, such that covered[i] is True iff
                    vertex i is covered.
    :param v: The vertex to cover its neighbours.
    :param t: The radius of the ball.
    :return: The updated list.
    """
    return [covered[i] or hamming_distance(i, v) <= t for i in range(len(covered))]


def ball_size(n: int, t: int):
    """
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :return: The number of neighbours with distance at most t from any vertex in the hamming graph of order n.
    """
    return sum(comb(n, i, exact=True) for i in range(t + 1))


def expected_steps(n: int, t: int):
    """
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :return: Expected number of steps to cover all the vertices in the hamming graph.
    """
    if n == t:
        return 1
    p = ball_size(n, t) / 2 ** n
    return -(n * math.log(2) + EulerGamma) / (math.log(1 - p)) + 0.5


def simulate_experiment(n: int, t: int):
    """
    Simulates an experiment with hamming graph of order n and ball with radius t.
    :param n: The order of the hamming graph.
    :param t: the radius of the ball that being covered at each step.
    :return: The number of steps it took to cover all the vertices in the graph.
    """
    covered = [False] * 2 ** n
    steps = 0

    while not all(covered):
        v = random.randint(0, 2 ** n - 1)
        covered = cover_ball(covered, v, t)
        steps += 1

    return steps


def repeat_experiment(n: int, t: int, runs: int):
    """
    Simulates an experiment multiple times and returns the expectation of the number of steps.
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :param runs: The number of repetitions.
    :return: The mean of the number of steps it took to cover all the vertices in the graph on each run.
    """
    results = [simulate_experiment(n, t) for _ in range(runs)]
    return np.mean(results)


def expectation_t_figure(n: int, runs: int):
    """
    Generates a plot of the expected number of steps to cover the graph as a function of the t.
    :param n: The order of the hamming graph.
    :param runs: The number of runs of each experiment.
    """
    t_values = range(n + 1)
    expectations = [repeat_experiment(n, t, runs) for t in t_values]

    plt.figure()
    plt.scatter(t_values, expectations, color='blue', alpha=0.6, label='Expected Number of Steps')
    plt.plot(t_values, expectations, color='blue', alpha=0.4)

    for t, y in zip(t_values, expectations):
        plt.annotate(f"{y:.2f}", (t, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    plt.xlabel("t")
    plt.ylabel("Expected Number of Steps")
    plt.title(f"n={n}, {runs} runs")
    plt.legend()

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))

    figure_name = f"expectation_graph_n={n}_runs={runs}.png"
    plt.savefig(figure_name)
    print(f"Figure saved as {figure_name}")


if __name__ == "__main__":
    expectation_t_figure(n=7, runs=1000)
