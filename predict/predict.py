#!/usr/bin/env python
# coding: utf-8
import boto3
import json
import sys
sys.path.append('../')
from train.load_data import load_data

app_name = 'cust-segm-api'
region = 'us-east-1'

if __name__ == '__main__':
    sm = boto3.client('sagemaker', region_name=region)
    smrt = boto3.client('runtime.sagemaker', region_name=region)

    # Check endpoint status
    endpoint = sm.describe_endpoint(EndpointName=app_name)
    print("Endpoint status: ", endpoint["EndpointStatus"])
    # Load preprocess objects
    dv, sc, pca, X_pca = load_data()
    # Predict from json
    json_path = 'customer.json'
    try:
        with open(json_path) as cust:
            json_dict = json.load(cust)

    except ValueError as e:
        logging.warning(f"Invalid JSON: {e}")

    data = dv.transform(json_dict)
    x_sc = sc.transform(data)
    x_pca = pca.transform(x_sc)
    payload = json.dumps(x_pca.tolist())
    prediction = smrt.invoke_endpoint(
        EndpointName=app_name,
        Body=payload,
        ContentType='application/json'
    )
    prediction = prediction['Body'].read().decode("ascii")
    print(prediction)
