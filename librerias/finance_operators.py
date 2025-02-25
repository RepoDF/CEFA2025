import numpy as np
import pandas as pd

def calculateLogReturns(df): 
    return np.log((df.shift(-1)/df).shift(1))

# Función que estima la volatilidad con el método EWMA
def calculateEWMAVol(log_returns, sigma_seed, lambda_scalar):
    # Calculate the 
    log_returns_squared = log_returns**2
    log_returns_squared.iloc[0] = sigma_seed**2
    log_returns_squared = log_returns_squared.ewm(alpha = 1-lambda_scalar, adjust= False).mean()
    log_returns_squared = np.sqrt(log_returns_squared)
    return log_returns_squared

def calculateEWMACov(log_returns, sigma_seed, lambda_scalar):
    # Calculate the 
    log_returns = log_returns.dropna()
    log_returns.iloc[0] = sigma_seed
    log_returns = log_returns.ewm(alpha = 1-lambda_scalar, adjust= False).mean()
    return log_returns