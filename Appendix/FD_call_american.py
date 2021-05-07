import numpy as np
import matplotlib.pyplot as plt

def call_payoff(strike, S_array):
    # create an array of strikes that is of the same size as the stock price array
    # len() is a functions that measures how big some array is
    length_of_S_array= len(S_array)
    strike_array=strike*np.ones(length_of_S_array)
    # now create an array of that same length (number of elements) that gives the
    # payoff values for a call option
    values=np.maximum(S_array-strike_array,0)
    return values


def set_parameters(S_0, T, r, sigma, div, N):
    # define the parameters as just outlined:

    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    nu = r - div - 0.5 * sigma ** 2

    pu = (1 / 2) * dt * ((sigma / dx) ** 2 + nu / dx)
    pm = 1 - dt * (sigma / dx) ** 2 - r * dt
    pd = (1 / 2) * dt * ((sigma / dx) ** 2 - nu / dx)

    # define the array of stock prices: it will be 2N+1 elements long, with S_0 as the middle value
    # each value up will be e^dx larger, while a value down will be e^(-dx) smaller

    S_array = S_0 * np.exp(dx * np.linspace(-N, N, 2 * N + 1))

    # this function gives as output the three probabilities and the 2N+1 array of stock prices.
    return pu, pm, pd, S_array


def step_back_call(pu, pm, pd, values, S_array):
    # note: no dependence on the step number n anymore, we follow the same procedure in each step!
    # we will need the stock price values, for the boundary conditions, so let's include those
    # as an argument of the function

    # first check how long your input array of values is
    length = len(values)

    # prepare the output array as one filled with zeros
    option_values = np.zeros(length)

    # now step back, first filling all elements except the top and bottom ones:
    option_values[1:length - 1] = pu * values[2:length] + pm * values[1:length - 1] \
                                  + pd * values[0:length - 2]

    # next, let us assign the bottom and top elements, consistent with the boundary conditions
    # for a call option

    option_values[0] = 0  # zero at the bottom
    option_values[length - 1] = option_values[length - 2] + S_array[length - 1] - S_array[length - 2]
    # call option value changes linearly with stock prices at the top

    return option_values


def American_call_grid(S_0, T, r, sigma, div, N, strike):
    # first set the parameters; note that this function gives a whole bunch of outputs at once:
    pu, pm, pd, S_array = set_parameters(S_0, T, r, sigma, div, N)

    # next, define the payoffs for the option at the end points, for the given strike price
    values = call_payoff(strike, S_array)

    # now use a for-loop to step back N times to produce the initial values
    for n in range(1, N + 1):
        # NEW: use our newly defined step back function for the whole grid! It is the
        # same each step, only the input values get updated
        intermediate_option_prices = step_back_call(pu, pm, pd, values, S_array)

        values = intermediate_option_prices  # update our values-array for the next step
        # we want to check for early exercise:
        # update the values with the early-exercise value if that gives a higher pay-off

        # NEW: we can do this for the entire array, rather than just the nodes in the tree
        values = np.maximum(values, call_payoff(strike, S_array))

    return S_array, values

if __name__ == "__main__":
    stockprice=100
    maturity=1
    interestrate=0.06
    volatility=0.2
    dividendrate=0.1
    steps=100
    strikeprice=100

    stocks, call_prices=American_call_grid(stockprice, maturity, interestrate, volatility, dividendrate, steps, strikeprice)

    print(stocks[steps], call_prices[steps])