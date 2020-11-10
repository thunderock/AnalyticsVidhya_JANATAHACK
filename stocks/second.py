import numpy as np
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from joblib import Parallel, delayed



train_file = 'Train.csv'
test_file = 'Test.csv'
submission_file = 'SampleSubmission.csv'
result_file = 'result.csv'


df = pd.read_csv(train_file, low_memory=False, parse_dates=['Date'])
tdf = pd.read_csv(test_file, low_memory=False, parse_dates=['Date'])
sub = pd.read_csv(submission_file, index_col='ID')


def do_stock(dframe):
    unpredictability_score = dframe['unpredictability_score'].unique()[0]
    stock = dframe['stock'].unique()[0]
    # date = dframe['Date']
    # id = dframe['ID']
    # close = dframe['Close']
    # print(id)

    df_extra = extract_features(dframe.drop(columns=['stock', 'unpredictability_score', 'Close']), column_id = 'ID',
                            column_sort='Date', show_warnings=False, impute_function=impute, disable_progressbar=True)

    df_extra['stock'] = stock
    df_extra['unpredictability_score'] = unpredictability_score
    dframe.set_index('ID', drop=False, inplace=True)
    df_extra['Date'] = dframe['Date']
    # print(df_extra['ID'])
    # print(df_extra['ID'])
    # df_extra['ID'] = dframe['ID']
    df_extra['close'] = dframe['Close']
    return df_extra

num_cores = 7


rows = Parallel(n_jobs=num_cores)(delayed(do_stock)(df[df['stock'] == i]) for i in tqdm(df.stock.unique(), total=len(df.stock.unique())))

# rows = [do_stock(df[df['stock'] == i]) for i in tqdm([1], total=1)]
# print(rows[0].ID)
df_extra = pd.concat(rows, axis=0)



print(df_extra.shape, df.shape)

df_extra.to_csv(result_file, index=False)
