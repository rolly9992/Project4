import pandas as pd 

df = pd.read_excel('data/sp500_tickerlist.xlsx')
df = df.sort_values('ticker',ascending=True)
print(df.columns)
print(df.ticker.tolist())