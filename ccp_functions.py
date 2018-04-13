import datetime as dt
import scipy.optimize as sco
import scipy.stats as scs
import statsmodels.regression.linear_model as sm

import pandas as pd
import pandas.tseries.offsets as pdtso

import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline


#%%

#==============================================================================
# TIME SERIES BASIC FUNCTIONS
#==============================================================================


# imports a time series from a given excel sheet and sorts the date index on the new_index
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

    # initialize the volatility constraint
    cons = [{'type':'ineq', 'fun':lambda x: target_vol - portfolio_stats(x, data_sliced, freq)[1]}]
    
    # initialize the weights constraint (weights should sum to max 100%)
    cons += [{'type':'eq', 'fun':lambda x: 1.0 - x.sum()}]

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
def signal_intensity(data_lagged, data_daily, date):
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
    
    # sign of the signal
    sign = True if (signal_value >= 0.0) else False
    
    # you want the signal to be either positive or negative, so you keep only the sign of interest
    data_daily_hist = pd.DataFrame(data_daily_hist)
    data_daily_hist = data_daily_hist[data_daily_hist >= 0.0] if sign else data_daily_hist[data_daily_hist < 0.0]
    
    # median value of the sign of interest
    data_daily_median = data_daily_hist.median().iloc[0]
    
    # if signal is positive, we want to see if it's over the median of positive values.
    # if signal is negative, we want to see if it's under the median of negative values.
    if sign:
        signal_intensity = 2.0 if (signal_value > data_daily_median) else 1.0
    else:
        signal_intensity = -2.0 if (signal_value < data_daily_median) else -1.0
    
    return signal_intensity

#%%

def signal_intensity2(data_lagged, data_daily, date):
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
    
    # sign of the signal
    sign = True if (signal_value >= 0.0) else False
    
    # you want the signal to be either positive or negative, so you keep only the sign of interest
    data_daily_hist = pd.DataFrame(data_daily_hist)
    data_daily_hist = data_daily_hist[data_daily_hist >= 0.0] if sign else data_daily_hist[data_daily_hist < 0.0]
    
    # median value of the sign of interest

    data_daily_Q1 = data_daily_hist.quantile(q=0.1).iloc[0]
    data_daily_Q2 = data_daily_hist.quantile(q=0.2).iloc[0]
    data_daily_Q3 = data_daily_hist.quantile(q=0.3).iloc[0]
    data_daily_Q4 = data_daily_hist.quantile(q=0.4).iloc[0]
    data_daily_median = data_daily_hist.median().iloc[0]
    data_daily_Q6 = data_daily_hist.quantile(q=0.6).iloc[0]
    data_daily_Q7 = data_daily_hist.quantile(q=0.7).iloc[0]
    data_daily_Q8 = data_daily_hist.quantile(q=0.8).iloc[0]
    data_daily_Q9 = data_daily_hist.quantile(q=0.9).iloc[0]
    
    # if signal is positive, we want to see if it's over the median of positive values.
    # if signal is negative, we want to see if it's under the median of negative values.
    if sign:
        if (signal_value > data_daily_median):
            signal_intensity2 = -1.0
        else:
            if (signal_value < data_daily_median & signal_value > data_daily_Q4):
                signal_intensity2 = -1.25
            else:
                if (signal_value < data_daily_Q4 & signal_value > data_daily_Q3):
                    signal_intensity2 = -1.5
                else:
                    if (signal_value < data_daily_Q3 & signal_value > data_daily_Q2):
                        signal_intensity2 = -1.75
                    else:
                        if (signal_value < data_daily_Q2 & signal_value > data_daily_Q1):
                            signal_intensity2 = -2.0
                        else:
                            if (signal_value < data_daily_Q1):
                                signal_intensity2 = -2.25
                
    else:
        if (signal_value < data_daily_median):
            signal_intensity2 = 1.00
        else:
            if (signal_value > data_daily_median & signal_value < data_daily_Q6):
                signal_intensity2 = 1.25
            else:
                if (signal_value > data_daily_Q6 & signal_value < data_daily_Q7):
                    signal_intensity2 = 1.5
                else:
                    if (signal_value > data_daily_Q7 & signal_value < data_daily_Q8):
                        signal_intensity2 = 1.75
                    else:
                        if (signal_value > data_daily_Q8 & signal_value < data_daily_Q9):
                            signal_intensity2 = 2
                        else:
                            if (signal_value > data_daily_Q9):
                                signal_intensity2 = 2.25
                                                  
    return signal_intensity2

#%%
def signal_intensity3(data_lagged, data_daily, date):
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
    
    # sign of the signal
    sign = True if (signal_value >= 0.0) else False
    
    # you want the signal to be either positive or negative, so you keep only the sign of interest
    data_daily_hist = pd.DataFrame(data_daily_hist)
    data_daily_hist = data_daily_hist[data_daily_hist >= 0.0] if sign else data_daily_hist[data_daily_hist < 0.0]
    
    # median value of the sign of interest

    data_daily_Q1 = data_daily_hist.mean().iloc[0]- 2*data_daily_hist.std().iloc[0]
    data_daily_Q2 = data_daily_hist.mean().iloc[0]-1.5*data_daily_hist.std().iloc[0]
    data_daily_Q3 = data_daily_hist.mean().iloc[0]-data_daily_hist.std().iloc[0]
    data_daily_Q4 = data_daily_hist.mean().iloc[0]-0.5*data_daily_hist.std().iloc[0]
    data_daily_Q5 = data_daily_hist.mean().iloc[0]
    data_daily_Q6 = data_daily_hist.mean().iloc[0]+0.5*data_daily_hist.std().iloc[0]
    data_daily_Q7 = data_daily_hist.mean().iloc[0]+data_daily_hist.std().iloc[0]
    data_daily_Q8 = data_daily_hist.mean().iloc[0]+1.5*data_daily_hist.std().iloc[0]
    data_daily_Q9 = data_daily_hist.mean().iloc[0]+2*data_daily_hist.std().iloc[0]


    # if signal is positive, we want to see if it's over the median of positive values.
    # if signal is negative, we want to see if it's under the median of negative values.
    if sign:
        if (signal_value > data_daily_Q5):
            signal_intensity3 = -1.0
        else:
            if (signal_value < data_daily_Q5 & signal_value > data_daily_Q4):
                signal_intensity3 = -1.25
            else:
                if (signal_value < data_daily_Q4 & signal_value > data_daily_Q3):
                    signal_intensity3 = -1.5
                else:
                    if (signal_value < data_daily_Q3 & signal_value > data_daily_Q2):
                        signal_intensity3 = -1.75
                    else:
                        if (signal_value < data_daily_Q2 & signal_value > data_daily_Q1):
                            signal_intensity3 = -2.0
                        else:
                            if (signal_value < data_daily_Q1):
                                signal_intensity3 = -2.25
                
    else:
        if (signal_value < data_daily_Q5):
            signal_intensity3 = 1.00
        else:
            if (signal_value > data_daily_Q5 & signal_value < data_daily_Q6):
                signal_intensity3 = 1.25
            else:
                if (signal_value > data_daily_Q6 & signal_value < data_daily_Q7):
                    signal_intensity3 = 1.5
                else:
                    if (signal_value > data_daily_Q7 & signal_value < data_daily_Q8):
                        signal_intensity3 = 1.75
                    else:
                        if (signal_value > data_daily_Q8 & signal_value < data_daily_Q9):
                            signal_intensity3 = 2
                        else:
                            if (signal_value > data_daily_Q9):
                                signal_intensity3 = 2.25
                                                  
    return signal_intensity3


#%%
def signal_intensity4(data_lagged, data_daily, date):
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
    
    # sign of the signal
    sign = True if (signal_value >= 0.0) else False
    
    # you want the signal to be either positive or negative, so you keep only the sign of interest
    data_daily_hist = pd.DataFrame(data_daily_hist)
    data_daily_hist = data_daily_hist[data_daily_hist >= 0.0] if sign else data_daily_hist[data_daily_hist < 0.0]
    
    # median value of the sign of interest

    data_daily_Q1 = 0.2
    data_daily_Q2 = 0.4
    data_daily_Q3 = 0.6
    data_daily_Q4 = 0.8
    data_daily_Q5 = 1
    data_daily_Q6 = 1.2
    data_daily_Q7 = 1.4
    data_daily_Q8 = 1.6
    data_daily_Q9 = 1.8
    
    #Compute the signal as the ratio comparared to historical mean or median
    signal_value=signal_value/data_daily_hist.mean().iloc[0]-1
    #signal_value=signal_value/data_daily_hist.median().iloc[0]-1

    # if signal is positive, we want to see if it's over the median of positive values.
    # if signal is negative, we want to see if it's under the median of negative values.
    if sign:
        if (signal_value > data_daily_Q5):
            signal_intensity4 = -1.0
        else:
            if (signal_value < data_daily_Q5 & signal_value > data_daily_Q4):
                signal_intensity4 = -1.25
            else:
                if (signal_value < data_daily_Q4 & signal_value > data_daily_Q3):
                    signal_intensity4 = -1.5
                else:
                    if (signal_value < data_daily_Q3 & signal_value > data_daily_Q2):
                        signal_intensity4 = -1.75
                    else:
                        if (signal_value < data_daily_Q2 & signal_value > data_daily_Q1):
                            signal_intensity4 = -2.0
                        else:
                            if (signal_value < data_daily_Q1):
                                signal_intensity4 = -2.25
                
    else:
        if (signal_value < data_daily_Q5):
            signal_intensity4 = 1.00
        else:
            if (signal_value > data_daily_Q5 & signal_value < data_daily_Q6):
                signal_intensity4 = 1.25
            else:
                if (signal_value > data_daily_Q6 & signal_value < data_daily_Q7):
                    signal_intensity4 = 1.5
                else:
                    if (signal_value > data_daily_Q7 & signal_value < data_daily_Q8):
                        signal_intensity4 = 1.75
                    else:
                        if (signal_value > data_daily_Q8 & signal_value < data_daily_Q9):
                            signal_intensity4 = 2
                        else:
                            if (signal_value > data_daily_Q9):
                                signal_intensity4 = 2.25
                                                  
    return signal_intensity4



#%% 
    
# gives the signal directions for an array of asset classes
#       col = "Monetary Policy"
#       signals_intensities(["Equities", "Bonds"], col)
# don't put RFR
def signal_directions(asset_classes, signal_name):
    # store the results
    signal_directions = []
    
    # find the signal
    signals_names = list(["Growth", "Inflation", "International Trade", "Monetary Policy", "Risk Sentiment"])
    signal_index = signals_names.index(signal_name)
    
    # define the relations for each asset class
    Equities_signals = list([1, -1, 1, -1, 1])
    Bonds_signals = list([-1, -1, -1, -1, -1])
    
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



# returns boundaries based on the signal
def signal_boundaries(intensity, directions):
    # number of assets
    n = len(directions)
    
    if abs(intensity) == 1:
        bounds = np.array([0.0, 0.5])
    elif abs(intensity) == 2:
        bounds = np.array([0.5, 1.0])
    
    # boundaries for the assets
    # we need a tuple (lb, ub) for each asset. and we must have lb < ub.
    # so we sort the list before converting it into a tuple
    signal_boundaries = [tuple(sorted(directions[i] * bounds * np.sign(intensity))) for i in range(n)]
    
    # additionnal boundary for the RFR
    signal_boundaries += [(-np.inf, np.inf)]
    
    return signal_boundaries



#%%

#%%


































