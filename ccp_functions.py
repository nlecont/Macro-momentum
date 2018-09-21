import datetime as dt
import scipy.optimize as sco
import scipy.stats as scs
import statsmodels.regression.linear_model as sm

import pandas as pd
import pandas.tseries.offsets as pdtso

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline


#%%

#==============================================================================
# TIME SERIES BASIC FUNCTIONS
#==============================================================================


# imports time series from a given excel sheet and sorts the date index on the new_index
def import_time_series(path, filename, sheet, new_index):
    # open the file
    TS = pd.read_excel(path + filename, sheet)
    
    # set the column dates as the index
    TS.set_index("Dates", inplace=True)
    
    # replaces the date index by the new_index
    TS = TS.reindex(index=new_index)
    
    # sort the dates in ascending order
    TS.sort_index(ascending=True, inplace=True)
    
    # forward fill the missing values.
    # Ex 1: weekend days will have the thursday value (forward filled)
    # Ex 2: for monthly data, all the month will be filled with the last value observed.
    TS.fillna(method="ffill", inplace=True)
    
    return TS

# merges two time series on a new index. Forward fills the missing dates, and removes the others
# This results as having one dataframe with the two dataframes TS_1 and TS_2 with new_index as index.
# TS_1 and TS_2 must have dates as indexes
def merge_time_series(TS_1, TS_2, new_index):
    # merges TS_1 and TS_2 on the outer product of their indexes.
    # Then replaces the merged index by the new_index.
    TS = TS_1.merge(TS_2, how="outer", left_index=True, right_index=True).reindex(index=new_index)
    
    # sort the dates in ascending order, so we can then fill the missing values
    TS.sort_index(ascending=True, inplace=True)
    
    # forward fill the missing values.
    # Ex 1: weekend days will have the thursday value (forward filled)
    # Ex 2: for monthly data, all the month will be filled with the last value observed.
    TS.fillna(method="ffill", inplace=True)
    
    # removes missing observations (e.g. if both TS_1 and TS_2 had no observation before a given date in new_index)
    TS.dropna(axis=0, how="all", inplace=True)
    
    return TS


#%%

#==============================================================================
# TIME SERIES MANIPULATION AND MOMENTUM INDICATORS
#==============================================================================

# returns the time series values shifted lag_days before
def shift_time_series(data, lag_days):
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_shifted = data.copy()
    
    # make sure this is a DataFrame
    data_shifted = pd.DataFrame(data_shifted)
    
    # returns the shifted index
    index_shifted = data_shifted.index.shift(n=-lag_days, freq='D')
    
    # shifts the values
    data_shifted = data_shifted.reindex(index=index_shifted)
    
    # reindexes with the original index
    data_shifted.set_index(data.index, inplace=True)
    
    return data_shifted


# uses shift_time_series to compute the difference between the data and the data_shifted
def diff_time_series(data, lag_days):
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_diff = data.copy()
    
    # make sure this is a DataFrame
    data_diff = pd.DataFrame(data_diff)
    
    # compute the shifted time series
    data_shifted = shift_time_series(data_diff, lag_days)

    # compute the shifted time series and returns the difference
    data_diff -= data_shifted
    
    return data_diff


# uses shift_time_series to compute the returns between the data and the data_shifted
def returns_time_series(data, lag_days):
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_returns = data.copy()
    
    # make sure this is a DataFrame
    data_returns = pd.DataFrame(data_returns)
    
    # compute the shifted time series
    data_shifted = shift_time_series(data_returns, lag_days)

    # returns the difference
    data_returns /= data_shifted - 1
    
    return data_returns * 100


# uses shift_time_series to compute the relative value of the time series compared to lag_days before
def relative_time_series(data, lag_days):
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_relative = data.copy()
    
    # make sure this is a DataFrame
    data_relative = pd.DataFrame(data_relative)
    
    # compute the shifted time series
    data_shifted = shift_time_series(data_relative, lag_days)

    # returns the relative value
    data_relative /= data_shifted
    
    return data_relative


# uses relative_time_series to compute the ratio of relative values
def ratio_relative_time_series(data, lag_days_numerator, lag_days_denominator, standardize):
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_ratio_relative = data.copy()
    
    # make sure this is a DataFrame
    data_ratio_relative = pd.DataFrame(data_ratio_relative)
    
    # compute the relative time series
    data_relative_numerator = relative_time_series(data_ratio_relative, lag_days_numerator)
    data_relative_denominator = relative_time_series(data_ratio_relative, lag_days_denominator)
    
    # if standardize, then we bring the denominator's relative increase on the same period of time than the numerator
    # Ex. numerator is 6 months (180 days), denominator is 3 years (1080 days), we want them to be comparable
    #       so we transform the denominator: denominator^(180/1080) <=> denominator^(0.5/3)
    #       and so both are expressed in terms of "6 months relative value"
    if standardize == True:
        exponent = lag_days_numerator / np.float(lag_days_denominator)
        data_relative_denominator = data_relative_denominator ** exponent
    
    # returns the difference
    data_ratio_relative = data_relative_numerator / data_relative_denominator
    
    return data_ratio_relative


# for a dataframe, returns a pd.Series of the row-wise geometric mean
def geometric_mean(data):
    # compute the geometric mean for each row, which is returned in a np.array
    res = scs.gmean(data, axis=1)
    
    # transforms the np.array into a pd.Series, with the corresponding dates.
    res = pd.Series(res, index=data.index, name="GMean")
    
    return res


# returns the time series of the moving average over lag_days 
def moving_average(data, lag_days):
    return data.rolling(window=lag_days, center=False).mean()


# returns a weighted moving average, here lag_days_array being an array of lag_days, each one being weighted by weights in the weights array.
# Ex: lag_days_array=[30, 365], weights=[3, 1]
# this will return: ( (30-days moving average) * 3 + (365-days moving average) * 1 ) / (3 + 1)
def weighted_moving_average(data, lag_days_array, weights):
    # first we create the results dataframe (same dimensions and index as data)
    data_result = data.copy()
    
    # set to zero, to be filled in the loop later
    data_result = pd.DataFrame(data_result) * 0.0
    
    # loop over the lag_days_array to compute the weighted moving average
    for i, lag_days in enumerate(lag_days_array):
        data_result += moving_average(data, lag_days) * np.float(weights[i]) / np.sum(weights)
        
    return data_result
    

# similar to weighted_moving_average, but here the periods are non overlapping: periods are one after the other
# (weighting from the earliest to the oldest period)
# Ex: lag_days_array=[30, 30, 30], weights=[3, 2, 1]
# will return a 90-days moving average, with the first third weighted by 3, the second third weighted by 2, and the last third weighted by 1
def decaying_moving_average(data, lag_days_array, weights):
    # first we create the results dataframe (same dimensions and index as data)
    data_result = data.copy()
    
    # set to zero, to be filled in the loop later
    data_result = pd.DataFrame(data_result) * 0.0
    
    # loop over the lag_days_array to compute the decaying moving average
    for i, lag_days in enumerate(lag_days_array):
        data_shifted = shift_time_series(data, 0 if i == 0 else np.sum(lag_days_array[:i]))
        
        data_result += moving_average(data_shifted, lag_days) * np.float(weights[i]) / np.sum(weights)
        
    return data_result



#%%

#==============================================================================
# PRE-REGRESSION DATA TREATMENT
#==============================================================================

# OLS regression of Y returns versus lagged X values, for a given frequency.
# Ex: X(t-1) will explain (Y(t)/Y(t-1) -1), t being the data frequency (e.g. 'D', 'W', 'M', 'Q'...)

###############

# Returns on the period [start_date, end_date], the time series from lag*freq periods before, at the given frequency
# Ex: if you want to do the regression on period [10,20], with a lag of 1 on the variable X,
#       you can lag by 1 so to have values of the period [9,19] indexed by period [10,20]
def data_lagged(data, start_date, end_date, freq, lag):
    # normal index = range for indexation
    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # shifted index = range for data
    index_shifted = index.shift(n=-np.int(lag), freq=freq)
    
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_lagged = data.copy()
    
    # reindexing the data to get values at the index_shifted
    data_lagged = data_lagged.reindex(index=index_shifted)
    
    # replacing the index with the non-lagged index
    data_lagged.set_index(index, inplace=True)
    
    return data_lagged

# Returns on the period [start_date, end_date], the time series of the "lag*freq" returns, at the given frequency
# Ex: if you want to do the regression on period [10,20], with a lag of 2 on the returns of Y,
#       you can lag by 2 so to have values of the period [8,20] to compute returns on period [10,20]
#       e.g. [8,10] to compute returns on 10, [9,11] to compute returns on 11, etc.
def data_returns(data, start_date, end_date, freq, lag):
    # we shift the start date because we compute returns, so we need "lag" more dates (before the period)
    start_date_shifted = pd.date_range(start_date, start_date).shift(n=-np.int(lag), freq=freq)[0]
    
    # total index = including all the values used to compute returns for period [start_date, end_date] 
    index = pd.date_range(start=start_date_shifted, end=end_date, freq=freq)
    
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_returns = data.copy()
    
    # reindexing the data with the index previously defined
    data_returns = data_returns.reindex(index=index)
    
    # computing Y returns for the given frequency, and over the period [start_date, end_date]
    data_returns = (data_returns / data_returns.shift(np.int(lag)) - 1).iloc[np.int(lag):] 
    
    return data_returns




#%%

#==============================================================================
# PORTFOLIO OPTIMIZATION
#==============================================================================


# returns a slice of the matrix of returns (preferably from function data_returns), using returns over "periods" before "date"
def data_slice(data, date, periods):
    # creating a copy not to modify the initial dataset (Params are indeed passed by reference)
    data_slice = data.copy()
    
    # get data until date
    # BE CAREFUL: date is the ACTUAL date, not the period date.
    # Ex: if data is monthly (every end of month = 31/01/2017) and you take first day of the month (01/01/2017),
    #       it won't take this month (last date = 31/12/2016)
    data_slice = data_slice.loc[:date]
    
    # take only the previous "periods" number of time series
    # this returns a dataframe with exactly "periods" number of times series
    data_slice = data_slice.iloc[-np.int(periods):]
    
    return data_slice


# returns the var_cov matrix from the matrix of returns (preferably from function data_returns), using returns over "periods" before "date"
def data_var_cov(data, date, freq, periods):
    # take the slice
    data_var_cov = data_slice(data, date, periods)
    
    # computes var_cov matrix and annualizes (based on the freq of the data) the VARIANCE (not volatility)
    return data_var_cov.cov() * annualization_factor(freq)


# factor to annualize returns / variance
def annualization_factor(freq):
    # annualize (based on the freq of the data)
    if freq == "D":
        factor = 252.0
    elif freq == "M":
        factor = 12.0
    elif freq == "Q":
        factor = 4.0
    else:
        factor = 1.0
        
    return factor


# function that computes return and vol of the portfolio on the time series data_sliced (preferably from function data_slice)
# returned stats are annualized using the given "freq"
# data_sliced is the investment universe, and weights must have the same size than data_sliced
def portfolio_stats(weights, data_sliced, freq):
    # make sure weights are in the appropriate format
    weights = np.array(weights)
    
    # compute the (annualized) RETURNS of the portfolio
    ptf_returns = np.sum(data_sliced.mean() * weights) * annualization_factor(freq)
    
    # compute the (annualized) VOLATILITY of the portfolio
    ptf_vol = np.sqrt(np.dot(weights.T, np.dot(data_sliced.cov(), weights)) * annualization_factor(freq))

    # returns portfolio annualized return and volatility
    return np.array([ptf_returns, ptf_vol])


# returns the optimized portfolio for a given target_vol, based on the characteristics of the dataset (data, date, freq, periods)
def portfolio_optimize(init_weights, target_vol, bnds, data, date, freq, periods):
    # make sure init_weights are in the appropriate format
    init_weights = np.array(init_weights)
    
    # get the returns time series
    data_sliced = data_slice(data, date, periods)
    
    # initialize the weights constraint (weights should sum to max 100%)
    cons = [{'type':'eq', 'fun':lambda x: 1.0 - x.sum()}]
    
    
    # maximize the sharpe ratio
    if target_vol == 'sharpe_ratio':
        
        risk_free_rate = data_sliced['RFR'].mean() * annualization_factor(freq)
    
        def objective(x):

            sharpe_ratio = (portfolio_stats(x, data_sliced, freq)[0] - risk_free_rate) / portfolio_stats(x, data_sliced, freq)[1]
            
            return -sharpe_ratio
    
        # optimization    
        opt_S = sco.minimize(objective, init_weights, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False})
        
    
    # maximizes utility function with a risk-aversion coefficient
    elif type(target_vol) == tuple:
        
        def objective(x):

            utility = portfolio_stats(x, data_sliced, freq)[0] - 0.5 * target_vol[1] * portfolio_stats(x, data_sliced, freq)[1] ** 2
            
            return -utility
        
        # optimization    
        opt_S = sco.minimize(objective, init_weights, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False})
        
        
    # Optimize with a volatility objective / constraint
    else:
        # initialize the volatility constraint
        cons += [{'type':'ineq', 'fun':lambda x: target_vol - portfolio_stats(x, data_sliced, freq)[1]}]
    
        # setting objective function
        def objective(x):
            # maximize return over the period..
            return -portfolio_stats(x, data_sliced, freq)[0]
    
        # optimization    
        opt_S = sco.minimize(objective, init_weights, method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False})
        
    
    # returns the optimal weights of the portfolio as a result
    return opt_S['x']


#%%

#==============================================================================
# SIGNAL TREATMENT
#==============================================================================

# gives the sign + intensity of the signal at a given date
# data_lagged, data_daily are pd.Series of the column of interest. Typically:
#       col = "Monetary Policy"
#       signal_intensity(X_macro[col], macro_data[col], "2017 11 30")
# returns the intensity of the signal for the "2017 11 30" (meaning we are forecasting the "2017 10 31")
def signal_intensity(data_lagged, data_daily, date, method='quantile', granularity=2, thresholds=[-2, 2]):
    # we will compare the signal value at the given date, to the median of whole historical data_daily (previous)
    # be careful, as data_lagged is shifted by a certain period, so we must not use the data_daily during that shift
    # so we need to reduce even more the data_daily
    
    # get the signal value at the given date (we suppose it exists as we won't use signal independently from other functions)
    signal_value = data_lagged.loc[date]
    
    # get the date from where we shifted
    date_before = data_lagged.loc[:date].index[-2]
    
    # get the time series of the signal (data_daily) until the date_before (i.e. all the historical data until we have to make the prevision)
    data_daily_hist = data_daily.copy()
    data_daily_hist = data_daily_hist.loc[:date_before].dropna()
    
    
    if method == 'quantile':
        signal_intensity = signal_intensity_quantiles(data_daily_hist, signal_value, granularity)
    elif method == 'zscore':
        signal_intensity = signal_intensity_zscores(data_daily_hist, signal_value, thresholds)
    elif method == 'zscore_excl':
        signal_intensity = signal_intensity_zscores(data_daily_hist, signal_value, thresholds, excl=True)
    elif method == 'zscore_robust':
        signal_intensity = signal_intensity_zscores(data_daily_hist, signal_value, thresholds, robust=True)
    else:
        raise ValueError('Method "%s" for computing signal intensity does not exist' % method)
    
    return signal_intensity


def signal_intensity_quantiles(data_daily_hist, signal_value, granularity):
    
    # sign of the signal
    sign = True if (signal_value >= 0.0) else False
    
    # you want the signal to be either positive or negative, so you keep only the sign of interest
    data_daily_hist = pd.DataFrame(data_daily_hist)
    data_daily_hist = data_daily_hist[data_daily_hist >= 0.0] if sign else data_daily_hist[data_daily_hist < 0.0]
    
    signal_intensity = 1 if sign else -1
    
    for i in range(1, granularity):
        # quantile value of the sign of interest
        data_daily_quantile = data_daily_hist.quantile(np.float64(i)/granularity).iloc[0]
        
        # if signal is positive, we want to see its quantile position. Same if its negative
        if sign:
            signal_intensity = i+1 if (signal_value > data_daily_quantile) else signal_intensity
        else:
            signal_intensity = -i-1 if (signal_value < data_daily_quantile) else signal_intensity
        
    return signal_intensity


def signal_intensity_zscores(data_daily_hist, signal_value, thresholds, excl=False, robust=False):
    
    # sign of the signal
    sign = True if (signal_value >= 0.0) else False
    
    signal_intensity = 0 if excl==True else (1 if sign else -1)
    
    if robust == False:
        signal_zscore = (signal_value - data_daily_hist.mean()) / data_daily_hist.std()
    else:
        signal_zscore = (signal_value - data_daily_hist.median()) / (data_daily_hist.quantile(0.75) - data_daily_hist.quantile(0.25))
    
    
    assert thresholds[0] < 0.0, ('Negative threshold for zscore (%0.2f) is positive' % thresholds[0])
    assert thresholds[1] > 0.0, ('Positive threshold for zscore (%0.2f) is negative' % thresholds[1])
    
    
    if signal_zscore > thresholds[1]:
        signal_intensity = 2
    elif signal_zscore < thresholds[0]:
        signal_intensity = -2  
    
    return signal_intensity


# gives the signal directions for an array of asset classes
#       col = "Monetary Policy"
#       signals_intensities(["Equities", "Bonds"], col)
# don't put RFR
def signal_directions(asset_classes, signal_name):
    # store the results
    signal_directions = []
    
    # find the signal
    signals_names = list(["Growth", "Inflation", "International Trade", "Monetary Policy", "Risk Sentiment",
                          "test"])
    signal_index = signals_names.index(signal_name)
    
    # define the relations for each asset class
    Equities_signals = list([1, -1, 1, -1, 1,
                             1])
    Bonds_signals    = list([-1, -1, -1, -1, -1,
                             1])
    
    # regroup for easier indexing
    assets_names = list(["Equities", "Bonds"])
    assets_signals = list([Equities_signals, Bonds_signals])
    
    # complete the directions for the given signal and the given assets
    for asset in asset_classes:
        # finds the index of the asset
        asset_index = assets_names.index(asset)
        
        # finds the signal direction for the given aset and the given signal_type
        signal_directions.append(assets_signals[asset_index][signal_index])
    
    # returns the array of signal directions
    return np.array(signal_directions)



# returns boundaries based on the signal and the considered granularity (intensity <= granularity)
def signal_boundaries(intensity, directions, granularity=2):
    # number of assets
    n = len(directions)
    
    granularity = np.float64(granularity)
    
    # upper & lower bounds based on the granularity
    ub = abs(intensity)/granularity
    lb = ub - 1/granularity
    bounds = np.array([lb, ub])
    
    # boundaries for the assets
    # we need a tuple (lb, ub) for each asset. and we must have lb < ub.
    # so we sort the list before converting it into a tuple
    signal_boundaries = [tuple(sorted(directions[i] * bounds * np.sign(intensity))) for i in range(n)]
    
    # additionnal boundary for the RFR
    signal_boundaries += [(-np.inf, np.inf)]
    
    return signal_boundaries





#%%

#==============================================================================
# RETURNS ANALYSIS
#==============================================================================

# Sharpe Ratio (the argument dataframe must have a RFR). Typically:
#       period_returns.columns = [["Equities", "Bonds", "RFR", "Strategy"]]
def sharpe_ratio(period_returns, freq):
    results = pd.Series(index=period_returns.columns, dtype=np.float64)
    
    for asset in period_returns:
        results[asset] = np.sqrt(annualization_factor(freq)) * (period_returns[asset] - period_returns["RFR"]).mean() / period_returns[asset].std()
        
    return results

def max_drawdown(returns):
    """Computes the max drawdown over an array of returns."""
    
    # computes prices from returns
    prices = (1.0 + returns).cumprod()
    
    # end of the period
    dd_end = np.argmax(np.maximum.accumulate(prices) - prices)
    
    # start of period
    dd_start = np.argmax(prices[:dd_end])
    
    # returns max_drawdown in percentage
    return (prices[dd_start] - prices[dd_end]) / prices[dd_start]


# computes the max drawdown over the period
def max_drawdown_period(period_returns):
    results = pd.Series(index=period_returns.columns, dtype=np.float64)
    
    for asset in period_returns:
        results[asset] = max_drawdown(period_returns[asset])
    
    return results

# returns a dataframe with basic statistics on the portfolio for the given period
def returns_analysis(strategy_returns, Y_assets, freq):
    
    # returns for the period (optimization_dates)
    period_returns = Y_assets.copy()
    period_returns = period_returns.reindex(index=strategy_returns.index)
    period_returns["Strategy"] = strategy_returns.copy()
    
    # statistics for the period
    period_statistics = pd.DataFrame(index=period_returns.columns)
    
    # return, vol, correlation, sharpe ratio
    period_statistics["Returns"] = (period_returns.mean() * annualization_factor(freq) * 100)
    period_statistics["Volatility"] = (np.sqrt(period_returns.var() * annualization_factor(freq)) * 100)
    period_statistics["Correlation"] = period_returns.corr()["Strategy"]
    period_statistics["Sharpe Ratio"] = sharpe_ratio(period_returns, freq)
    period_statistics["Drawdown"] = max_drawdown_period(period_returns)
    # CAN ADD OTHER STATISTICS, IN THIS CASE MUST MODIFY "names_indicators" IN THE FUNCTION "strategy_analysis" BELOW
    
    return period_statistics


def strategy_analysis(periods, strategy_results, Y_assets, freq):
    """Returns a MultiIndex DataFrame with statistics of the strategy
    for different optimization periods.
    
    periods -- list of tuples
    strategy_results -- list of DataFrames from function 'optimization'
    Y_assets -- DataFrame from function 'data_returns'
    """
    
    names_indicators = ["Returns",
                        "Volatility",
                        "Correlation",
                        "Sharpe Ratio",
                        "Drawdown"]
    
    names_periods = period_names_list(periods)
    
    names_columns = pd.MultiIndex.from_product([names_periods, names_indicators], names=['Periods', 'Indicators'])
    names_index = [Y_assets.columns.tolist() + ["Strategy"]]
    my_df = pd.DataFrame(index=names_index, columns=names_columns)
    
    
    for i, period_results in enumerate(strategy_results):
        my_df[names_periods[i]] = returns_analysis(period_results["Return"], Y_assets, freq)
        
    return my_df
    # HOW TO USE THIS DATAFRAME
    # my_df.sort_index(axis=1).loc(axis=1)[:, 'Volatility']

#%%


























