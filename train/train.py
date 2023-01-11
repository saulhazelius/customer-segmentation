#!/usr/bin/env python
# coding: utf-8
import mlflow.sklearn
from sklearn.cluster import KMeans
from load_data import load_data

if __name__ == "__main__":
    mlflow.set_experiment("customer-segmentation-kmeans")
    with mlflow.start_run(run_name="kmeans-n-4") as run:

        dv, sc, pca, X_pca = load_data()
        
        model = KMeans(n_clusters=4)
        model.fit(X_pca)

        mlflow.sklearn.log_model(model, "cust-segm-model")
        mlflow.end_run()
