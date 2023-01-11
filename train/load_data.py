#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data():

    df = pd.read_csv("../data/marketing_campaign.csv", delimiter=";")
    df.drop(columns=["Z_CostContact", "Z_Revenue"], inplace=True)
    df["Dt_Customer"] = pd.to_datetime(df.Dt_Customer)
    df["Dt_days"] = df["Dt_Customer"].apply(
        lambda x: np.floor((datetime.today() - x) / np.timedelta64(1, "D"))
    )

    # Remove outliers of Year_birth and Income:
    # year birth quantiles
    q1_year = df.Year_Birth.quantile(0.05)
    q3_year = df.Year_Birth.quantile(0.95)
    iqr_year = q3_year - q1_year

    # Income quantiles
    q1_inc = df.Income.quantile(0.05)
    q3_inc = df.Income.quantile(0.95)
    iqr_inc = q3_inc - q1_inc

    # Limits
    low_year = q1_year - 1.5 * iqr_year
    high_year = q3_year + 1.5 * iqr_year
    
    low_inc = q1_inc - 1.5 * iqr_inc
    high_inc = q3_inc + 1.5 * iqr_inc

    df = df[(df.Year_Birth > low_year) & (df.Year_Birth < high_year)]
    df = df[(df.Income > low_inc) & (df.Income < high_inc)]
    df.reset_index(inplace=True, drop=True)

    # Now remove unnecesary columns
    df.drop(columns=["ID", "Dt_Customer"], inplace=True)

    # PCA 3 dims, without features of promotions (last kmeans experiment in EDA.ipynb):
    cols = [col for col in df.columns if "Accepted" in col] + [
        "Response",
        "Complain",
    ]

    df.drop(columns=cols, inplace=True)
    df_dict = df.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    dv.fit(df_dict)
    X_feats = dv.transform(df_dict)

    sc = StandardScaler()
    sc.fit(X_feats)
    X_std = sc.transform(X_feats)
    pca = PCA(n_components=3)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)

    return dv, sc, pca, X_pca