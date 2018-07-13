import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb




def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "learning_rate": 0.1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

def add_columns(df):
    df['title_des'] = df['title'] + df['description']
    df['price'] = np.log1p(df['price'])
    df['region_city'] = df['region'] + df['city']
    df = df.drop(['region','city','title','description'],axis=1)
    df['item_seq_number'] = np.log1p(df['item_seq_number'])
    return df


def prepare(train_df, test_df):
    train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
    test_df["activation_weekday"] = test_df["activation_date"].dt.weekday
    cat_vars = ["region_city","title_des","parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
    for col in cat_vars:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    cols_to_drop = ["item_id", "user_id", "activation_date", "image"]
    train_X = train_df.drop(
        cols_to_drop + ["deal_probability"],
        axis=1)
    test_X = test_df.drop(cols_to_drop, axis=1)
    return train_X, test_X

if "__name__" == "__main__":
    train_df = pd.read_csv("train.csv", parse_dates=["activation_date"])
    test_df = pd.read_csv("test.csv", parse_dates=["activation_date"])
    train_y = train_df["deal_probability"].values
    test_id = test_df["item_id"].values
    train_df = add_columns(train_df)
    test_df = add_columns(test_df)
    train_X, test_X= prepare(train_df,test_df)
    dev_X = train_X.iloc[:-200000,:]
    val_X = train_X.iloc[-200000:,:]
    dev_y = train_y[:-200000]
    val_y = train_y[-200000:]

    # Training the model #
    pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

    # Making a submission file #
    pred_test[pred_test>1] = 1
    pred_test[pred_test<0] = 0
    sub_df = pd.DataFrame({"item_id":test_id})
    sub_df["deal_probability"] = pred_test
    sub_df.to_csv("result1_lgb.csv", index=False)


