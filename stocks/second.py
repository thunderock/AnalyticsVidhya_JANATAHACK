import numpy as np
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from joblib import Parallel, delayed



train_file = 'super_train.csv'
result_file = 'result.csv'


df = pd.read_csv(train_file, low_memory=False, parse_dates=['Date'])



def do_stock(dframe):
    unpredictability_score = dframe['unpredictability_score'].unique()[0]
    stock = dframe['stock'].unique()[0]
    df_extra = extract_features(dframe.drop(columns=['stock', 'unpredictability_score', 'Close', 'High', 'Low', 'Open']), column_id = 'ID',
                            column_sort='Date', show_warnings=False, impute_function=impute, disable_progressbar=True,
                            n_jobs=0)

    df_extra['stock'] = stock
    df_extra['unpredictability_score'] = unpredictability_score
    # dframe.set_index('ID', drop=False, inplace=True)
    df_extra['Date'] = dframe['Date']
    df_extra['Close'] = dframe['Close']
    df_extra['High'] = dframe['High']
    df_extra['Low'] = dframe['Low']
    df_extra['Open'] = dframe['Open']
    return df_extra

num_cores = 7


# rows = Parallel(n_jobs=num_cores)(delayed(do_stock)(df[df['stock'] == i]) for i in tqdm([1], total=1))
rows = Parallel(n_jobs=num_cores)(delayed(do_stock)(df[df['stock'] == i]) for i in tqdm(df.stock.unique(), total=len(df.stock.unique())))

del df
df_extra = pd.concat(rows, axis=0)



print(df_extra.shape, df.shape)

df_extra.to_csv(result_file)
