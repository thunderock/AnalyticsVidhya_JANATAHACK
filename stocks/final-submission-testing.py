import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import lightgbm as lgb
import warnings
from joblib import Parallel, delayed
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
warnings.filterwarnings('ignore')



train_file = 'training_files/result_{}.csv'
super_train = 'super_train.csv'


stock = 1
df = pd.read_csv(train_file.format(stock), index_col='ID', parse_dates=['Date'])

tdf = pd.read_csv(super_train, index_col='ID')


df = df.join(tdf[['Open_hat', 'High_hat', 'Low_hat', 'Close_hat']])

def expand_df(dframe):
    dFrame = dframe.copy()
    dFrame['day'] = dFrame.Date.apply(lambda x: x.day)
    dFrame['month'] = dFrame.Date.apply(lambda x: x.month)
    dFrame['year'] = dFrame.Date.apply(lambda x: x.year)
    dFrame['dayofweek'] = dFrame.Date.apply(lambda x: x.dayofweek)
    dFrame['dayofyear'] = dFrame.Date.apply(lambda x: x.dayofyear)
    dFrame['weekofyear'] = dFrame.Date.apply(lambda x: x.weekofyear)
    dFrame['year_diff'] = dFrame.Date.apply(lambda x: x.year - 2017)
    dFrame['days_so_far'] = dFrame.Date.apply(lambda x: (x - pd.Timestamp('2017-01-03')).days)

    return dFrame



df = expand_df(df)



cat_cols = [
    'holiday',
    'stock',
    'day',
     'month',
     'year',
     'dayofweek',
     'dayofyear',
     'weekofyear',
    'year_diff',
    'unpredictability_score']
excluded_cols = ['Close_hat', 'Open_hat', 'High_hat', 'Low_hat']





def rolled_mean(dframe):
    dframe['ID'] = dframe.index

    dFrame = dframe.copy()
    dFrame = roll_time_series(dFrame, show_warnings=False, disable_progressbar=True, column_id='stock', column_sort='Date', max_timeshift=30, min_timeshift=0)
    for col in excluded_cols:
        dframe[col + '_roll_mean'] = dframe['ID'].apply(lambda x: dFrame[dFrame['ID'] == x][col].mean())
    return dframe.set_index('ID', drop=True)


print(df.shape)




print(df.columns.tolist())

encoder = LabelEncoder()
for col in tqdm(cat_cols):
    df[col] = encoder.fit_transform(df[col])


df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]



X, y = df[df['Close'].notna()].drop(columns=['Close', 'Open', 'High', 'Low'], axis=1), df[['Close', 'stock']][df['Close'].notna()]
X_test, y_test = df[df['Close'].isna()].drop(columns=['Close', 'Open', 'High', 'Low'], axis=1), df[['Close', 'stock']][df['Close'].isna()]


model_store1 = [0] * 103
metrics1 = [0] * 103
df = None

grid = {'learning_rate': [.1, .2, .3, .4], 'depth': [1, 2, 3, 4, 5], 'iterations': [200, 400, 600, 800, 1000]}
preds1 = []

# for stock in tqdm(range(103)):
def get_predictions(stock):
    df = pd.read_csv(train_file.format(stock), index_col='ID', parse_dates=['Date'])
    df = df.join(tdf[['Open_hat', 'High_hat', 'Low_hat', 'Close_hat']])
    df = expand_df(df)

    df = rolled_mean(df)

    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]

    X, y = df[df['Close'].notna()].drop(columns=['Close', 'Open', 'High', 'Low'], axis=1), df[['Close', 'stock']][df['Close'].notna()]
    X_test, y_test = df[df['Close'].isna()].drop(columns=['Close', 'Open', 'High', 'Low'], axis=1), df[['Close', 'stock']][df['Close'].isna()]

    # X, y = df[df['Close'].notna()].drop(columns=['Close', 'Open', 'High', 'Low'], axis=1), df[['Close', 'stock']][df['Close'].notna()]
    # X_test, y_test = df[df['Close'].isna()].drop(columns=['Close', 'Open', 'High', 'Low'], axis=1), df[['Close', 'stock']][df['Close'].isna()]

    # print(X_test.columns.tolist())
    X_tr, X_val, y_tr, y_val = train_test_split(X, y['Close'], train_size=.8, random_state=11568)
    model_store1[stock] = CatBoostRegressor(loss_function='RMSE', depth=2, learning_rate=0.4, iterations=800,
        random_seed=18,
        od_type='Iter',
        od_wait=20,
        thread_count=1 # task_type="GPU"
    )
#     print(X_tr.columns)
    model_store1[stock].fit(
        X_tr, y_tr, use_best_model=True,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        verbose=False,
        plot=False,
    )

    return pd.DataFrame({'ID': X_test.index, 'Close': model_store1[stock].predict(X_test)})


num_cores = 7
preds1 = Parallel(n_jobs=num_cores)(delayed(get_predictions)(stock) for stock in tqdm(range(103)))




pd.concat(preds1).to_csv('result.csv', index=False)
print(pd.concat(preds1).shape)
