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
result_file = 'result_{}.csv'


df = pd.read_csv(train_file, low_memory=False, parse_dates=['Date'])



def do_stock(dframe):
    f_cols = ['stock', 'unpredictability_score', 'Close', 'High', 'Low', 'Open', 'holiday']
    df_extra = extract_features(dframe.drop(columns=f_cols),
                            column_id='ID',column_sort='Date',
                            show_warnings=False, impute_function=impute,
                            disable_progressbar=True, n_jobs=1)
    X, y = df_extra[df_extra.index.isin(dframe[dframe.Close.notna()]['ID'])], df_extra.join(dframe.set_index('ID')['Close'])['Close'][pd.notna(df_extra.join(dframe.set_index('ID')['Close'])['Close'])]

    dx = select_features(X, y, n_jobs=1)
    dframe[f_cols + ['ID', 'Date']].set_index('ID', drop=True).join(df_extra[dx.columns]).to_csv(result_file.format(dframe.stock.unique()[0]))


    return True

num_cores = 2


rows = Parallel(n_jobs=num_cores)(delayed(do_stock)(df[df['stock'] == i]) for i in tqdm(df.stock.unique(), total=len(df.stock.unique())))
