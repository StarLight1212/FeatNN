import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, mean_absolute_percentage_error


def rmse(y,f):
    """
    Task:    To compute root mean squared error (RMSE)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rmse   RSME
    """

    rmse = sqrt(((y - f)**2).mean(axis=0))

    return rmse


def pearson(y,f):
    """
    Task:    To compute Pearson correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rp     Pearson correlation coefficient
    """

    rp = np.corrcoef(y, f)[0,1]

    return rp


def spearman(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient

     Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rs     Spearman's rank correlation coefficient
    """

    rs = stats.spearmanr(y, f)[0]

    return rs


def r_square_score(y, pred):
    return r2_score(y, pred)


def MedAE(y, pred):
    return median_absolute_error(y, pred)


def MAE(y, pred):
    return mean_absolute_error(y, pred)


def MAPE(y, pred):
    return mean_absolute_percentage_error(y,pred)