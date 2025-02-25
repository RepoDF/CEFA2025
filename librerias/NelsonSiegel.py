import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def estimateNelsonSiegel(df: pd.DataFrame, lambda_grid=None):
    """
    Estimate the Nelson-Siegel model using linear regression and grid search for lambda.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'Maturity' and 'Yield' columns.
    lambda_grid (array-like, optional): Grid of lambda values to search. Default is np.concatenate([np.logspace(-3, 1, 50), [1.37, 3]]).
    
    Returns:
    dict: Best lambda, estimated parameters (beta0, beta1, beta2), and best RMSE.
    """
    if lambda_grid is None:
        lambda_grid = np.concatenate([np.logspace(-3, 1, 50), [1.37, 3]])
    
    maturities = df["Maturity"].values
    yields = df["Yield"].values
    
    best_lambda = None
    best_rmse = float('inf')
    best_params = None
    
    for lambd in lambda_grid:
        # Compute explanatory variables
        X1 = (1 - np.exp(-lambd * maturities)) / (lambd * maturities)
        X2 = X1 - np.exp(-lambd * maturities)
        X = np.column_stack([np.ones_like(maturities), X1, X2])  # Add intercept
        
        # Solve linear regression
        model = LinearRegression(fit_intercept=False)
        model.fit(X, yields)
        y_pred = model.predict(X)
        
        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(yields, y_pred))
        
        # Update best lambda if RMSE improves
        if rmse < best_rmse:
            best_rmse = rmse
            best_lambda = lambd
            best_params = model.coef_
    
    return {
        "best_lambda": best_lambda,
        "beta0": best_params[0],
        "beta1": best_params[1],
        "beta2": best_params[2],
        "best_rmse": best_rmse
    }

# Example usage:
data = pd.DataFrame({
    "Maturity": [0.25, 0.5, 1, 2, 5, 10, 20, 30],
    "Yield": [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.052]
})

result = estimateNelsonSiegel(data)
print(result)

def standardizeYieldSeries(yield_series):
    yield_df = yield_series.to_frame()
    yield_df.columns = ['Yield']
    yield_df.loc['9 Mo',:] = np.nan
    yield_df.loc['4 Yr',:] = np.nan
    yield_df.loc['6 Yr',:] = np.nan
    yield_df.loc['8 Yr',:] = np.nan
    yield_df = yield_df.reset_index().rename(columns = {'index': 'TimePeriod'})
    yield_df['Yield'] = yield_df['Yield']/100
    
    yield_df['TimePeriodInMonths'] = yield_df['TimePeriod'].apply(period_to_months)
    yield_df = yield_df.sort_values('TimePeriodInMonths')
    yield_df['Maturity'] = yield_df['TimePeriodInMonths']/12
    # If you don't want to keep the 'TimePeriodInMonths' column, you can drop it
    yield_df = yield_df.reset_index(drop=True)
    return yield_df
