import pandas as pd
import streamlit as st
import numpy as np
import pickle
import yfinance as yf
from plotly import graph_objs as go

from sklearn.preprocessing import MinMaxScaler

with open(r"C:\Users\LENOVO\Project\Stock Market Prediction forecasting\model_pickle",'rb') as f:
    model = pickle.load(f)


def load_data():
    data = pd.read_csv(r"C:\Users\LENOVO\Project\Stock Market Prediction forecasting\finaldata.csv")
    lclose = pd.DataFrame(data.Close)
    test = lclose[-365:]
    test = np.array(test)
    return test
    

def forecast(n):
    forecastn_day = np.array([])
    sample = load_data()
    scaler = MinMaxScaler(feature_range = (0,1))
    sample = scaler.fit_transform(sample)
    sample = sample.reshape(-1,1)
    
    for i in range(n):
        pred = model.predict(sample)
        last_pred = pred[-1]
        forecastn_day = np.append(forecastn_day, last_pred, axis = 0)
        sample = np.append(sample, [last_pred], axis = 0)
    forecastn_day = scaler.inverse_transform([forecastn_day])
    return forecastn_day


def adjustments(forecast_value, n_days):
    df = pd.DataFrame(forecast_value)
    dfn = pd.DataFrame(df.T.values, columns = ['Predictions'])
    start_date = '2024-03-01'
    end_date = '2024-03-30'
    date_range = pd.date_range(start=start_date, end=end_date)

    # Assign the generated date range as the index of the DataFrame
    dfn.index = date_range[:n_days]
    dfn.index = dfn.index.date
    return dfn


def march_graph(pred, n):
    start = '2024-03-01'
    end = '2024-03-30'
    stock = 'AMZN'

    latest_data = yf.download(stock, start, end)

    close_col = pd.DataFrame(latest_data.Close)
    close_col.index = close_col.index.date
    close_col.reset_index(inplace = True)
    close_col['index'] = pd.to_datetime(close_col['index'])
    close_col.set_index('index', inplace = True)
    #st.write(close_col.info())

    close_col = close_col.resample('D').mean()
    close_col = close_col.interpolate('linear')
    size = len(close_col)
    df = pred[:size]
    df.reset_index(inplace = True)

    actual = close_col[:n]
    actual.reset_index(inplace = True)
    actual['Predictions'] = df['Predictions']
    actual.set_index('index', inplace = True)
    actual.index = actual.index.date
    st.write(actual)

    graph = actual.copy()
    graph.reset_index(inplace = True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = graph['index'], y = graph.Close, name = 'Actual data'))
    fig.add_trace(go.Scatter(x = graph['index'], y = graph.Predictions, name = 'Predicted data'))
    fig.layout.update(title_text = 'Comparison')
    st.plotly_chart(fig)


def main():
    st.title('Amazon stock price predictions')
    st.header('For upto 30 days')
    n_days = st.slider('Number of days for which you want to make the prediction', 1,30)

    forecast_value = forecast(n_days)
    result = adjustments(forecast_value, n_days)
    heading = 'Forecast for '+ str(n_days)+ ' day(s)'
    st.subheader(heading)
    st.write(result)
    if st.button('Compare with actual March\'s data'):
        march_graph(result, n_days)

if __name__ == '__main__':
    main()
