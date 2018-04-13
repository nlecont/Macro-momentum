import datetime as dt
import scipy.optimize as sco
import scipy.stats as scs
import statsmodels.regression.linear_model as sm

import pandas as pd
import pandas.tseries.offsets as pdtso

import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline


path = "G:/M2/CCP/Data/"


################################
# Largest continuous time index
first_date, last_date = "1969 01 31", "2018 01 01"
dtindex = pd.date_range(start=first_date, end=last_date, freq='D')


#%%

#==============================================================================
# ASSET CLASSES
#==============================================================================

asset_classes = pd.DataFrame(index=dtindex)
asset_classes.sort_index(ascending=True, inplace=True)

asset_classes["Equities"] = import_time_series(path, "asset classes.xlsx", "S&P index", dtindex)["SPXT"]
asset_classes["Bonds"] = import_time_series(path, "asset classes.xlsx", "Barclays index", dtindex)["Barclays"]


# ALWAYS PUT THE RISK-FREE RATE AT THE END
asset_classes["RFR"] = import_time_series(path, "asset classes.xlsx", "Risk-free asset", dtindex)["RFA"]



#%%

#==============================================================================
# RESET MACRO DATA
#==============================================================================

# here we will store our macro momentum indicators
macro_data = pd.DataFrame(index=dtindex)
macro_data.sort_index(ascending=True, inplace=True)


#%%

#==============================================================================
# MACRO DATA V1
#==============================================================================


################################
# Monetary policy

# Monetary policy trends are captured using one-year changes in the front end of the yield curve.
# From 1992 onwards, I use two-year yields, while prior to 1992 I use Libor and its international equivalents.

policy = pd.DataFrame(index=dtindex)

policy["USGG2YR"] = import_time_series(path, "policy.xlsx", "USGG2YR", dtindex)
policy["FEDL01"] = import_time_series(path, "asset classes.xlsx", "Risk-free asset", dtindex)["FEDL01"]

# Computing YoY changes
policy["2Y YoY"] = policy["USGG2YR"] - policy["USGG2YR"].shift(365)
policy["2Y YoY"] = policy["2Y YoY"].loc["19920101":] # take only after 1992
policy["FF YoY"] = policy["FEDL01"] - policy["FEDL01"].shift(365)
policy["FF YoY"] = policy["FF YoY"].loc[:"19911231"] # take only before 1992
policy.fillna(0, inplace=True) # fill NAs with 0 to allow sum (next line)

macro_data["Monetary Policy"] = policy["2Y YoY"] + policy["FF YoY"] # continuous YoY changes
macro_data["Monetary Policy"] = macro_data[macro_data.index > "1970 01 30"]["Monetary Policy"]

################################
# International trade

# International trade trends are captured using one-year changes in spot exchange rates against an export-weighted basket.

macro_data["International Trade"] = import_time_series(path, "currencies.xlsx", "DXY", dtindex)
macro_data["International Trade"] = (macro_data["International Trade"].shift(365) / macro_data["International Trade"] - 1)


################################
# Risk sentiment

# Changes in risk sentiment are captured using one-year equity market excess returns.

sentiment = pd.DataFrame(index=dtindex)

sentiment["SPXT"] = import_time_series(path, "asset classes.xlsx", "S&P index", dtindex)["SPXT"]
sentiment["RFA"] = import_time_series(path, "asset classes.xlsx", "Risk-free asset", dtindex)["RFA"]

macro_data["Risk Sentiment"] = (sentiment["SPXT"]/sentiment["SPXT"].shift(365) - 1) - (sentiment["RFA"]/sentiment["RFA"].shift(365) - 1)


################################
# Business cycle

# Business cycle trends are captured using one-year changes in forecasts of real GDP growth and CPI inflation.
# From 1990 onward forecast data is from Consensus Economics.
# Prior to 1990, I use one-year changes in realized year-on-year real GDP growth and CPI inflation, lagged one quarter

# 1. GDP Growth

macro_data["Growth"] = import_time_series(path, "cycle.xlsx", "GDPG", dtindex)
macro_data["Growth"] = macro_data["Growth"] - macro_data["Growth"].shift(365)

# 2. Inflation

macro_data["Inflation"] = import_time_series(path, "cycle.xlsx", "CPI_F", dtindex)
macro_data["Inflation"] = macro_data["Inflation"] - macro_data["Inflation"].shift(365)



#%%

#==============================================================================
# MACRO DATA V2
#==============================================================================


################################
# Monetary policy

monetary_policy = import_time_series(path, "policy.xlsx", "ChicagoFedIndex", dtindex)

macro_data["Monetary Policy"] = moving_average(diff_time_series(monetary_policy, 7), 30) / moving_average(diff_time_series(monetary_policy, 7), 90)


################################
# International trade

# Dollar Index
international_trade = import_time_series(path, "currencies.xlsx", "DXY", dtindex)

# our trade indicator is the ratio: change in the last 6 months / average semiannual change in the last 3 years.
macro_data["International Trade"] = ratio_relative_time_series(international_trade, 183, 1096, True) * 100.0


################################
# Risk sentiment

risk_sentiment = pd.DataFrame(index=dtindex)

risk_sentiment["USDJPY"] = import_time_series(path, "sentiment.xlsx", "USDJPY", dtindex)
risk_sentiment["TED Spread"] = import_time_series(path, "sentiment.xlsx", "TED Spread", dtindex)

risk_sentiment["GMean"] = geometric_mean(risk_sentiment)

macro_data["Risk Sentiment"] = decaying_moving_average(risk_sentiment["GMean"], [30, 30, 30], [1.0, 0.5, 0.33])


################################
# Business cycle

business_cycle = pd.DataFrame(index=dtindex)

business_cycle["PMI"] = import_time_series(path, "cycle.xlsx", "PMI", dtindex)
business_cycle["Consumers"] = import_time_series(path, "cycle.xlsx", "ConsExp", dtindex)
# business_cycle["CPI_F"] = import_time_series(path, "cycle.xlsx", "CPI_F", dtindex)

business_cycle["GMean"] = geometric_mean(business_cycle)

macro_data["Business Cycle"] = ratio_relative_time_series(business_cycle["GMean"], 183, 1096, True) * 100.0




#%%


"""
to do:
    
paper side:
    1. 
    2. find more data for monetary policy (as in the paper)
    3. try again to find some kind of GDP forecast, or build our own

our side:
    1. solve the pb of negative data when computing GMean (inflation...)
    2. pb of centering-reducing => we use forward data...
    3.

"""


#%%

#==============================================================================
# OLD TEST OF CUTTING QUANTILES
#==============================================================================

###############
# REGRESSIONS #
###############


start_date, end_date = "2005 01 01", "2018 01 01"

Y_assets = data_returns(asset_classes, start_date, end_date, "M", 1)
X_macro = data_lagged(macro_data[["Monetary Policy", "International Trade", "Business Cycle"]], start_date, end_date, "M", 1)



# regression on each asset class
for asset in Y_assets.columns:
    res = sm.OLS(Y_assets[asset], sm.add_constant(X_macro)).fit()
    print(res.summary())




#%%

# argument = pd.Series
def cut_quantile(data, lbound, ubound):
    data_cut = data.copy()
    
    # select data only above lbound quantile and beyond ubound quantile
    data_cut = data_cut[(data_cut < data_cut.quantile(lbound)) | (data_cut > data_cut.quantile(ubound))]
    
    return data_cut

# for each indicator we cut the data at the required quantile, and apply a simple OLS regression of assets on the indicator
for indicator in X_macro.columns:
    
    X_macro_signal = cut_quantile(X_macro[indicator], 0.0, 0.75)
    Y_assets_signal = Y_assets.reindex(X_macro_signal.index)
    
    for asset in Y_assets.columns:
        
        res = sm.OLS(Y_assets_signal[asset], X_macro_signal).fit()
        print(res.summary())
        

#%%
X_macro_signal = pd.DataFrame(index=X_macro.index)
for indicator in X_macro.columns:
    
    X_macro_signal[indicator] = cut_quantile(X_macro[indicator], 0.0, 0.5)

X_macro_signal.dropna(axis=0, how="any", inplace=True)

Y_assets_signal = Y_assets.reindex(X_macro_signal.index)


for asset in Y_assets.columns:
    
    res = sm.OLS(Y_assets_signal[asset], X_macro_signal).fit()
    print(res.summary())

































