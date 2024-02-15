import pandas as pd
import yfinance as yf
import datetime as dt
from darts.models import*
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.metrics import mape, mase
import logging
import warnings
from warnings import filterwarnings


def oracle(portfolio, start_date, weights=None, prediction_days=None, based_on='Close'):


  print("Collecting datas...")
  #define weights
  if weights == None:
    weights = [1.0 / len(portfolio)] * len(portfolio)


  #define today
  today = dt.datetime.today().strftime('%Y-%m-%d')


  #clean output from warning
  logger = logging.getLogger()
  warnings.simplefilter(action='ignore', category=FutureWarning)
  filterwarnings('ignore')
  
  logging.disable(logging.INFO)


  mape_df = pd.DataFrame()
  mape_df = mape_df._append({'Ai Model' : 0 }, 
                ignore_index = True)

  final_df = pd.DataFrame()
  final_df = final_df._append({'Ai Model' : 0 },
                ignore_index = True)

  for asset in portfolio:

    result = pd.DataFrame()
    df = yf.download(asset, start=start_date, end=today, progress=False)["Close"]
    df = pd.DataFrame(df)
    df.reset_index(level=0, inplace=True)

    if prediction_days==None:
      x = 1
      while x/(len(df)+x) < 0.3:
        x+=1
        prediction_days = x

    def eval_model(model):
      model.fit(train)
      forecast = model.predict(len(val))
      result[model] = [mape(val, forecast)]

    prediction = pd.DataFrame()


    def predict(model):
      model.fit(train)
      forecast = model.predict(len(val))
      pred = model.predict(prediction_days)
      b = [str(pred[-1][0][0])][0]
      b = b.split("array([[[")
      c = b[1].split("]]])")
      d = c[0][ : -3] 
      b = float(d)
      prediction[model] = [str(round(((b-start_value)/start_value)*100,3))+' %']

    series = TimeSeries.from_dataframe(df, 'Date', based_on, freq='D')
    series = fill_missing_values(series)

    train_index = round(len(df.index)*0.7)
    train_date = df.loc[[train_index]]['Date'].values
    date = str(train_date[0])[:10]
    date = date.replace('-', '') 
    timestamp = date+'000000'
    train, val = series.split_before(pd.Timestamp(timestamp))

    print("Evaluating the models for "+str(asset)+"...")

    eval_model(ExponentialSmoothing())
    

    print("Models evaluated!")

    result.columns = ['Ai Model']
    result.index = [asset]
    mape_df = pd.concat([result, mape_df])
    start_pred = str(df["Date"].iloc[-1])[:10]
    start_value = df[based_on].iloc[-1]
    start_pred = start_pred.replace('-', '') 
    timestamp = start_pred+'000000'
    train, val = series.split_before(pd.Timestamp(timestamp))
    
    print("Making the predictions for "+str(asset)+"...")
    predict(ExponentialSmoothing())
    
    print("Predictions generated!")

    prediction.columns = ['Ai Model']
    prediction.index = [asset]
    final_df = pd.concat([prediction, final_df])

  print("\n")
  print("Assets MAPE (accuracy score)")
  with pd.option_context('display.max_rows', None, 'display.max_columns', None) and pd.option_context('expand_frame_repr', False):
    print(mape_df.iloc[:-1,:])
  mape_df = pd.DataFrame(mape_df.iloc[:-1,:])
  print("\n")
  print("Assets returns prediction for the next "+str(prediction_days)+" days")
  with pd.option_context('display.max_rows', None, 'display.max_columns', None) and pd.option_context('expand_frame_repr', False):
    print(final_df.iloc[:-1,:])
  final_df = pd.DataFrame(final_df.iloc[:-1,:])
  
  portfolio_pred = pd.DataFrame()

  for column in final_df.columns:
    rets = []
    for index in final_df.index:
      place = portfolio.index(index)
      returns = float(final_df[column][index][:-1])
      wts = weights[portfolio.index(index)]
      ret = (returns*wts)
      rets.append(ret)
    portfolio_pred[column] = rets
    portfolio_pred[column] = portfolio_pred[column].sum()

  print("\n")
  print("Portfolio returns prediction for the next "+str(prediction_days)+" days")
  print(portfolio_pred.iloc[0])
  

  logger.disabled = False