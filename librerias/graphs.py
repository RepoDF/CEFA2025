import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates

def plot_yield_curve(df):
    df['Y'] = round(df['Yield']*100,4)
    df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
    df['N'] = round(df['NS']*100,4)
    df2 = df.copy()
    df2 = df2.style.format({'Maturity': '{:,.2f}'.format,'Y': '{:,.2%}', 'N': '{:,.2%}'})
    import matplotlib.pyplot as plt
    import matplotlib.markers as mk
    import matplotlib.ticker as mtick
    plt.style.use('dark_background')
    fontsize=15
    fig = plt.figure(figsize=(13,7))
    ax = plt.axes()
    X = df["Maturity"]
    Y = df["Y"]
    x = df["Maturity"]
    y = df["N"]
    ax.plot(x, y, color="#F5C74D", label="NS") #orange
    plt.scatter(x, y, marker="o", c="#F5C74D") #orange
    plt.scatter(X, Y, marker="o", c="#103CC8") #blue
    plt.xlabel('Period',fontsize=fontsize)
    plt.ylabel('Interest',fontsize=fontsize)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="lower right", title="Yield")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.title("Nelson-Siegel Model - Fitted Yield Curve",fontsize=fontsize)
    plt.show()

#plot_yield_curve(yield_df)

def heatmap(corr):
    fig, ax = plt.subplots(figsize=(9,9))
    orig_map=plt.cm.get_cmap('RdYlGn',256)
    off_diag_mask = np.eye(*corr.shape, dtype=bool)
    ax = sns.heatmap(
        corr.round(1), 
        annot=True,
        vmin=-1, vmax=1, center=0,
        cmap='Accent',
        square=True,
        cbar=False,
        mask=~off_diag_mask
    )
    ax = sns.heatmap(
        corr.round(1), 
        annot=True,
        vmin=-1, vmax=1, center=0,
        cmap=orig_map.reversed(),
        square=True,
        cbar=False,
        mask=off_diag_mask
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right',
        fontsize=10
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=10
    )
    plt.title("Matriz de Correlaciones", weight='bold')
    
    return fig

def calculate_simple_moving_average(close, n):
    """
    Calculates the simple moving average.
    
    Parameters
    ----------
    Close : DataFrame
        Close prece for every ticker and date
    
    n : int
        window size for the moving average computation

    Returns
    -------
    sma : DataFrame
      Simple movig average
    
    """
    sma = close.rolling(window=n).mean()
    return sma

def calculate_simple_moving_sample_stdev(close, n):
    """
    Calculates the simple moving standard deviation.
    
    Parameters
    ----------
    Close : DataFrame
        Close prece for every ticker and date
    
    n : int
        window size for the moving average computation

    Returns
    -------
    sma : DataFrame
      Simple movig standard deviation
    
    """
    smsd = close.rolling(window=n).std()
    return smsd

def create_bollinger_band_signal_extensive(close, n):
    """
    Create a meanreverting-based signal based on the upper and lower 
    bands of the Bollinger bands. Geenerate a buy sigal when the price 
    is bellow the lower band and a sell signal when the price is above
    the uper band.

    Parameters
    ----------
    Close : DataFrame
        Close price for every ticker and date
    
    n: int
       window size for the moving average and standard deviation computation
    
    Returns
    -------
    Signals : DataFrame
      Buy (1) Sell (-1) or do nothing signal (0)
    """
    sma = calculate_simple_moving_average(close, n)
    stdev = calculate_simple_moving_sample_stdev(close, n)
    upper = sma + 2 * stdev
    lower = sma - 2 * stdev

    sell = close > upper
    buy = close < lower

    signal = 1 * buy - 1 * sell

    return signal, sma, upper, lower

def plot_bollinger_bands(df, column='Close', window=20):
    df.index = pd.to_datetime(df.index)

    """
    Plots Bollinger Bands with buy/sell signals and improved x-axis readability.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    
    column : str
        Column name of the price data.
    
    window : int
        Rolling window size.

    Returns
    -------
    None
    """
    df = df.copy()
    
    # Generate signals and Bollinger Bands
    signal, sma, upper, lower = create_bollinger_band_signal_extensive(df[column], window)
    
    # Plot price and Bollinger Bands
    plt.figure(figsize=(12,6), dpi=100)  # Increase DPI for clarity
    plt.plot(df.index, df[column], label='Precio', color='blue', alpha=0.6)
    plt.plot(df.index, sma, label=f'{window}-Días SMA', color='black', linestyle='dashed')
    plt.fill_between(df.index, upper, lower, color='gray', alpha=0.2, label='Bollinger Bands')

    # Plot buy/sell signals
    plt.scatter(df.index[signal == 1], df[column][signal == 1], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(df.index[signal == -1], df[column][signal == -1], label='Sell Signal', marker='v', color='red', alpha=1)

    # Improve x-axis formatting
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto spacing
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability

    plt.legend()
    plt.title(f'Bandas de Bollinger ({window}-Días) con señales de trading')
    plt.xlabel('Fecha')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a light grid
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

def plot_time_series(series, title="Gráfico de Series de Tiempo", xlabel="Fecha", ylabel="Precio", 
                     date_format='%Y-%m-%d', tick_interval=None, figsize=(10, 5)):
    """
    Plots a time series with formatted datetime x-axis and returns the figure object.
    
    Parameters:
    - series: pandas Series with a datetime index
    - title: str, title of the plot
    - xlabel: str, label for x-axis
    - ylabel: str, label for y-axis
    - date_format: str, format for date labels (default: '%Y-%m-%d')
    - tick_interval: int, optional, interval for showing x-axis ticks
    - figsize: tuple, figure size
    
    Returns:
    - fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(ax=ax, title=title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    # Format date labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    
    # Set tick interval if provided
    if tick_interval:
        ax.set_xticks(series.index[::tick_interval])
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    return fig  # Return the figure object

