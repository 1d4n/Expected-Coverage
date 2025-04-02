import math
import multiprocessing
import random
from os import path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.special import comb
from scipy.stats import linregress
from sympy import EulerGamma

FIGURES_PATH = "figures"


def hamming_distance(u: int, v: int):
    """
    Calculates the hamming distance between two vectors.
    :param u: First vector.
    :param v: Second vector.
    :return: Hamming distance between the two vectors.
    """
    return bin(u ^ v).count('1')


def ball_size(n: int, t: int):
    """
    Calculates the size of a t-radius ball of a vertex in the hamming graph.
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :return: The number of vertices in the ball.
    """
    return sum(comb(n, i, exact=True) for i in range(t + 1))


def get_ball(n: int, t: int, v: int):
    """
    Returns the t-radius ball of the vertex v.
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :param v: The vertex to find its ball.
    :return: A set of the vertices in the ball.
    """
    return {u for u in range(2 ** n) if hamming_distance(u, v) <= t}


def cover_ball(covered: list[bool], n: int, t: int, v: int):
    """
    Covers the vertices in the t-radius ball of the vertex v.
    :param covered: List of vertices that are already covered. Its length is 2**n, such that covered[v] is True iff
                    vertex v is covered.
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :param v: The vertex to cover its neighbours.
    """
    for u in get_ball(n, t, v):
        covered[u] = True


def get_new_covered(not_covered: set[int], t: int, v: int):
    """
    Returns the vertices that are in the t-radius ball of the vertex v, such that they have not been covered yet.
    :param not_covered: Set of vertices that have not been covered yet.
    :param t: Radius of the ball.
    :param v: The vertex to cover its neighbours.
    :return: Set of the new-covered vertices.
    """
    new_covered = set()
    for u in not_covered:
        if hamming_distance(u, v) <= t:
            new_covered.add(u)
    return new_covered


def expected_steps(n: int, t: int):
    """
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :return: Approximation of the expected number of steps to cover all the vertices in the hamming graph.
    """
    if n == t:
        return 1
    p = ball_size(n, t) / 2 ** n
    return -(n * math.log(2) + EulerGamma) / (math.log(1 - p)) + 0.5


def simulate_experiment(n: int, t: int):
    """
    Simulates an experiment with hamming graph of order n and ball with radius t.
    :param n: The order of the hamming graph.
    :param t: The radius of the ball that being covered at each step.
    :return: The number of steps it took to cover all the vertices in the graph.
    """
    covered = [False] * 2 ** n
    steps = 0
    while not all(covered):
        v = random.randint(0, 2 ** n - 1)
        cover_ball(covered, n, t, v)
        steps += 1
    return steps


def simulate_experiment_large_t(n: int, t: int):
    """
        Same as simulate_experiment, but more efficient for larger t.
    """
    not_covered = {v for v in range(2 ** n)}
    steps = 0
    while not_covered:
        v = random.randint(0, 2 ** n - 1)
        not_covered -= get_new_covered(not_covered, t, v)
        steps += 1
    return steps


def repeat_experiment(n: int, t: int, runs: int):
    """
    Simulates an experiment multiple times and returns the average number of steps it took to cover all the vertices
    in the graph.
    :param n: The order of the hamming graph.
    :param t: The radius of the ball.
    :param runs: The number of repetitions.
    :return: The average of the number of steps it took to cover all the vertices in the graph.
    """
    params = [(n, t)] * runs
    with multiprocessing.Pool() as pool:
        results = pool.starmap(simulate_experiment, params)
    print(f"n={n}, t={t}, avg={np.mean(results)}")
    return np.mean(results)


def generate_figure(filename, title, x_values, y_values, x_label, y_label, points_label):
    """
    Generates a figure.
    :param filename: The filename to save the figure with.
    :param title: Title of the figure.
    :param x_values: Values for the x-axis.
    :param y_values: Values for the y-axies.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param points_label: Label for the points (their meaning).
    """
    plt.figure()
    plt.scatter(x_values, y_values, color='blue', alpha=0.6, label=points_label)
    plt.plot(x_values, y_values, color='blue', alpha=0.4)

    linear_equation = linear_regression(x_values, y_values, x_label)
    exp_equation = exponential_regression(x_values, y_values, x_label)
    log_equation = logistic_regression(x_values, y_values, x_label)
    plt.plot([], [], ' ', label=linear_equation)
    plt.plot([], [], ' ', label=exp_equation)
    plt.plot([], [], ' ', label=log_equation)

    for x, y in zip(x_values, y_values):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))

    figure_path = path.join(FIGURES_PATH, filename)
    plt.savefig(figure_path)
    print(f"Figure saved as {filename}")


def expectation_t_figure(n: int, runs: int):
    """
    Generates a plot of the average number of steps to cover the graph as a function of t.
    :param n: The order of the hamming graph.
    :param runs: The number of times to run each experiment.
    """
    t_values = range(1, n + 1)
    mean_values = [repeat_experiment(n, t, runs) for t in t_values]
    generate_figure(filename=f"n={n}_runs={runs}.png", title=f"n={n}, {runs} runs", x_values=t_values,
                    y_values=mean_values, x_label='t', y_label='Average Steps', points_label='Average Number of Steps')


def expectation_n_figure(n_values, runs, t):
    """
    Generates a plot of the average number of steps to cover the graph as a function of n.
    :param n_values: The values of n (x-axis).
    :param runs: The number of times to run each experiment.
    :param t: A function of n, such than t.get(n) is the radius of the ball.
    """
    mean_values = [repeat_experiment(n, t.get(n), runs) for n in n_values]
    generate_figure(filename=f"{t}_runs={runs}.png", title=f"{t}, {runs} runs", x_values=n_values, y_values=mean_values,
                    x_label='n', y_label='Average Steps', points_label='Average Number of Steps')


class DivT:
    def __init__(self, k):
        self.k = k

    def __str__(self):
        return f"t=div(n,{self.k})"

    def get(self, n):
        return n // self.k


class ConstT:
    def __init__(self, k):
        self.k = k

    def __str__(self):
        return f"t={self.k}"

    def get(self, n):
        return self.k


class MinusT:
    def __init__(self, k):
        self.k = k

    def __str__(self):
        return f"t=n-{self.k}"

    def get(self, n):
        return n - self.k


def linear_regression(x_values, y_values, x_name):
    res = linregress(x_values, y_values)
    return f"linear:  y = {res.slope:.2f}{x_name} {"+" if res.intercept > 0 else "-"} {abs(res.intercept):.2f}"


def exponential_regression(x_values, y_values, x_name):
    a, b = np.polyfit(x_values, np.log2(y_values), 1)
    return f"exp:  y = {2 ** b:.2f} * 2**({a:.2f}{x_name})"


def logistic_regression(x_values, y_values, x_name):
    a, b = np.polyfit(np.log2(x_values), y_values, 1)
    return f"log:  y = {a:.2f} * log2({x_name}) {"+" if b > 0 else "-"} {abs(b):.2f}"


if __name__ == "__main__":
    RUNS = 1000
    expectation_n_figure(n_values=range(2, 21, 2), runs=RUNS, t=DivT(2))  # t=n/2
    expectation_n_figure(n_values=range(13), runs=RUNS, t=ConstT(1))  # t=n-1
    expectation_t_figure(n=12, runs=RUNS)
