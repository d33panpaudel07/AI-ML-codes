import math
import numpy as np
import pandas as pd


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# hlo
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def euclidean_distance(x1, x2):
    """Calculates the l2 distance between two vectors"""
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


# can also implemented using numpy

# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))

# day 11
