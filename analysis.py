import pandas as pd
import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def nn_function():
    df = pd.read_csv("analysis.csv")
    # probably the clumsiest way to split the dataframe
    df.drop(df.columns[[1, 2]], axis=1)
    if len(df.index) < 280: # score 25 in training is usually enough
        print("Not enough values for training")
        return None
    npIn = np.array([[]])
    dfIn = pd.DataFrame(columns=('x1', 'x2'))
    dfOut = pd.DataFrame(columns=('x3',))
    for i in range(int(df[df.columns[[0]]].max())):
        df1 = df[df['win_lose'] == i]
        if len(df1.index) >= 3:
            df1 = df1.tail(3)
            nar = df1['x']
            narIn = nar.head(2)
            dfIn.loc[i] = narIn.values

            narOut = nar.tail(1)
            dfOut.loc[i] = narOut.values

    # input dataset
    X = dfIn.values/500

    # output dataset
    y = dfOut.values/500

    # seed random numbers
    np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = 2 * np.random.random((2, 1)) - 1

    for iter in range(10000):
        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0, syn0))

        l1_error = y - l1

        # multiply the error by the
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1, True)

        # update weights
        syn0 += np.dot(l0.T, l1_delta)

    return syn0

# at this stage syn0 is our "brain" or the ability to predict the outpur given the input