#We want to create a python file that will calculate the Black-Scholes value given some information

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm

def ExactCallSolution(r, sigma, Strike, MaxTime, MaxPrice, MinPrice, PriceSteps, Option, Graph):

    OptionPrice = np.zeros((PriceSteps,MaxTime))
    SigmaSquared = sigma**2
    StockPrice = np.linspace(MinPrice, MaxPrice, PriceSteps)
    Tau = np.linspace(0, 1, MaxTime)  # This is the transformed time space
    [Tau, S] = np.meshgrid(Tau, StockPrice)
    x = np.log(S/Strike) + (r - 0.5*SigmaSquared)*Tau
    d1 = (x + SigmaSquared * Tau) / (sigma * np.sqrt(Tau))
    d2 = (x) / (sigma * np.sqrt(Tau))

    if Option == 'Call':
        OptionPrice = Strike*np.exp(x + 0.5*SigmaSquared*Tau)*norm.cdf(d1) - Strike*norm.cdf(d2)

    elif Option == 'Put':
        OptionPrice = -Strike * np.exp(x + 0.5 * SigmaSquared * Tau) * norm.cdf(-d1) + Strike * norm.cdf(-d2)

    if Graph == 'True':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Tau, S, OptionPrice, cmap=cm.viridis)
        ax.set_xlabel('Time to Expiration')
        ax.set_ylabel('Price of Underlying Asset')
        ax.set_zlabel('Option Price')
        ax.set_title('Exact Solution of an Option')
        plt.show()

    return OptionPrice

def L2Error(Estimated, Exact, PriceSteps, MaxPrice, MinPrice):
    h = (MaxPrice - MinPrice)/PriceSteps
    Error = np.sqrt(h*np.nansum((Estimated - Exact)**2))
    return Error

def ExplicitOption(r, sigma, Strike, MaxTime, MaxPrice, MinPrice, PriceSteps, Option, Graph):

    OptionPrice = np.zeros((PriceSteps,MaxTime)) #Here we create the array to store option prices
    StockPrice = np.linspace(MinPrice, MaxPrice, PriceSteps)
    TimeStep = (1)/MaxTime
    Tau =  np.linspace(0, 1, MaxTime) #This is the transformed time space

    #We now need to enforce boundary conditions.
    np.seterr(divide='ignore') # We get divide by zero errors due to the Tau's below

    if Option == 'Call':
        OptionPrice[:,0] = np.maximum(StockPrice - Strike, 0)
        OptionPrice[0,:] = 0
        OptionPrice[-1,:] = MaxPrice - Strike*np.exp(-r*Tau)
        x = np.log(MaxPrice/Strike) + (r - 0.5*sigma**2)*Tau
        d1 =  (x + (sigma**2)*Tau)/(sigma*np.sqrt(Tau))
        d2 =  (x)/(sigma*np.sqrt(Tau))
        OptionPrice[-1, :] = Strike*np.exp(x + (0.5*sigma**2)*Tau)*norm.cdf(d1) - Strike*norm.cdf(d2)

    elif Option == 'Put':
        OptionPrice[:,0] = np.maximum(0, Strike - StockPrice)
        x = np.log(MinPrice/Strike) + (r - 0.5*sigma**2)*Tau
        d1 =  (x + (sigma**2)*Tau)/(sigma*np.sqrt(Tau))
        d2 =  (x)/(sigma*np.sqrt(Tau))
        OptionPrice[0, :] = - Strike*np.exp(x + (0.5*sigma**2)*Tau)*norm.cdf(-d1) + Strike*norm.cdf(-d2)

    SigmaSquare = sigma**2
    k = TimeStep

    #Here we can actually implement the scheme
    for j in np.arange(MaxTime-1):
        for i in np.arange(1,PriceSteps-1):
            OptionPrice[i,j+1] = (0.5*k*(SigmaSquare*i**2 - r*i ))*OptionPrice[i-1,j] + (1 - k*SigmaSquare*i**2)*OptionPrice[i,j] + 0.5*k*(SigmaSquare*i**2 + r*i )*OptionPrice[i+1,j]

    #We generate the plot using MatPlotLib

    [Tau2, S] = np.meshgrid(Tau, StockPrice)
    if Graph == 'True':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Tau2, S, OptionPrice, cmap=cm.viridis)
        ax.set_xlabel('Time to Expiration')
        ax.set_ylabel('Price of Underlying Asset')
        ax.set_zlabel('Option Price')
        ax.set_title('Explicit Scheme Solution of an Option')
        plt.show()

    ExactSoln = ExactCallSolution(r, sigma, Strike, MaxTime, MaxPrice, MinPrice, PriceSteps, Option, Graph)
    Error = L2Error(OptionPrice, ExactSoln, PriceSteps, MaxPrice, MinPrice)

   # return OptionPrice, ExactSoln, Error

ExplicitOption(0.1, 0.05, 10, 100, 20, 0, 200, 'Put', Graph = 'False')