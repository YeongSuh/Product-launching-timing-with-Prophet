# 1) Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# 2) Load raw datasets of keywords and relevant data
bank_etf = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2023_SS_Business Forecasting/Final Project/bank_etf.csv")
bank_stocks = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2023_SS_Business Forecasting/Final Project/bank_stocks.csv")
dividend_growth = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2023_SS_Business Forecasting/Final Project/dividend_growth.csv")
dividend_stock = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2023_SS_Business Forecasting/Final Project/dividend_stock.csv")
saving_interst = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2023_SS_Business Forecasting/Final Project/installment_interest_rate.csv")
interst_rate = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2023_SS_Business Forecasting/Final Project/interest_rate.csv")
kodex_bank = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/2023_SS_Business Forecasting/Final Project/KODEX_Bank.csv", thousands = ',')

# 3) Change the date format into "datetime" in each raw data
data = [bank_etf, bank_stocks, dividend_growth, dividend_stock, saving_interst, interst_rate, kodex_bank]
for i in data:
    i["Date"] = pd.to_datetime(i["Date"])

# 4) Simple Analysis on the competing product (KODEX Bank)
kodex_bank["Balance"].fillna(method='ffill', inplace=True)
plt.scatter(kodex_bank["Date"], kodex_bank['Balance'], label='Kodex Bank Balance')
plt.legend(loc='upper left')
plt.title("KODEX BANK Balance Status")
plt.show()

# 5) Visualize the trend of the interest rate of Korea
plt.plot(interst_rate['Date'], interst_rate['Interest_rate'], label='Interest rate')
plt.title('Interest Rate of Korea')
plt.show()

# 6) Simple plot of keywords data
plt.figure(figsize=(10, 6))
plt.scatter(bank_etf['Date'], bank_etf['bank_etf'], s=1, c='b', label='bank etf')
plt.scatter(bank_stocks['Date'], bank_stocks['bank_stocks'], s=1, c='r', label='bank stocks')
plt.scatter(dividend_growth['Date'], dividend_growth['dividend_growth'], s=1, c='g', label='dividend growth')
plt.scatter(dividend_stock['Date'], dividend_stock['dividend_stocks'], s=1, c='y', label='dividend stock')
plt.legend(loc='upper left')
plt.show()

# 7) Change the column names
bank_etf.columns = ['Date', 'bank_etf']

# 8) Merge all raw data into a new dataframe
merge_1 = pd.merge(left=bank_etf, right=bank_stocks, how='outer', on='Date')
merge_2 = pd.merge(left=merge_1, right=dividend_growth, how='outer', on='Date')
merge_3 = pd.merge(left=merge_2, right=dividend_stock, how='outer', on='Date')
merge_4 = pd.merge(left=merge_3, right=interst_rate, how='outer', on='Date')
merge_5 = pd.merge(left=merge_4, right=kodex_bank, how='outer', on='Date')
merge_5 = merge_5.sort_values("Date")
merge_5['Interest_rate'].fillna(method='ffill', inplace=True)
merge_5['Balance'].fillna(method='ffill', inplace=True)

# 9)Assgin merged dataset into a new variable and remove the row with the data in 2017
dat = merge_5[(merge_5['Date'] >= "2018-02-26") & (merge_5["Date"] < "2023-02-27")]

# 10) Check the correlation of bank_etf with other variables (Heatmap)
dat_corr = dat.corr()
sns.heatmap(dat_corr, annot=True, fmt=".2f",
            mask=~np.tri(dat_corr.shape[0], k=-1, dtype=bool),
            cbar=False)

# 11) Analysis of each keyword with Prophet (Only wrote the codes for 'bank_etf' keyword, as the structures are similar with other keywords)
model = Prophet()
bank_etf.columns = ['ds', 'y']
bank_etf = bank_etf[bank_etf['y'] < 70].  # Remove outliers
m_etf = model.fit(bank_etf)
future = m_etf.make_future_dataframe(periods=365 * 2)
forecast = m_etf.predict(future)
fig = m_etf.plot(forecast, xlabel='Date', ylabel='bank_etf')
plt.show()
fig2 = m_etf.plot_components(forecast)
plt.show()

# 12) Predicting "ETF Balance" with an Additional Regressor in Prophet
columns_order = ["Date", "Balance", "Interest_rate", "bank_etf", "bank_stocks", "dividend_growth", "dividend_stocks"]
dat = dat[columns_order]
dat.columns = ["ds", "y", "Interest_rate", "bank_etf", "bank_stocks", "dividend_growth", "dividend_stocks"]

model = Prophet()
model.add_regressor(name='Interest_rate')
model.add_regressor(name='bank_etf')
model.add_regressor(name='bank_stocks')
model.add_regressor(name='dividend_growth')
model.add_regressor(name='dividend_stocks')

from datetime import timedelta

# Remove final 2 weeks of training data
train = dat[dat['ds'] < dat['ds'].max() - timedelta(weeks=2)]

m2 = model.fit(train)

future = m2.make_future_dataframe(periods=14)
future['Interest_rate'] = dat['Interest_rate']
future['bank_etf'] = dat['bank_etf']
future['bank_stocks'] = dat['bank_stocks']
future['dividend_growth'] = dat['dividend_growth']
future['dividend_stocks'] = dat['dividend_stocks']

forecast = m2.predict(future)

fig3 = m2.plot_components(forecast)
plt.show()

# 13) Analyze the coefficients of each variables 
from prophet.utilities import regressor_coefficients
regressor_coefficients(model)

# 14) Forecast Evaluation
from prophet.plot import add_changepoints_to_plot

model = Prophet(uncertainty_samples=1000, yearly_seasonality=4)
model.fit(dat)
future = model.make_future_dataframe(periods=365 * 2)
forecast = model.predict(future)
fig = model.plot(forecast)
add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()
    
