import json
import boto3
import os
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import requests
from urllib.parse import unquote_plus
from emg_processor import process_emg_signal

s3 = boto3.client('s3')

# AppSync config
APPSYNC_API_URL = os.environ.get("APPSYNC_API_URL")
APPSYNC_API_KEY = os.environ.get("APPSYNC_API_KEY")

def lambda_handler(event, context):
    # 1. Extract S3 object info
    record = event['Records'][0]['s3']
    bucket = record['bucket']['name']
    key = unquote_plus(record['object']['key'])
    print(f"Triggered on key: {key}")

    # Filter out irrelevant keys
    if '/emg/' not in key or '/processed/' in key or not key.endswith('.csv'):
        print("Skipping file not in /emg/ or already processed.")
        return {'statusCode': 200, 'body': 'Skipped'}

    filename = key.split("/")[-1]
    session_id = filename.replace(".csv", "")

    # 2. Read and clean CSV
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    df.columns = df.columns.str.strip().str.lower()
    raw_emg = pd.to_numeric(df['emg_value'], errors='coerce').dropna().values

    # 3. Process signal
    processed_df, metrics = process_emg_signal(raw_emg)

    # 4. Generate new S3 path for processed file
    parts = key.split('/')
    identity_id = parts[1]   # e.g. us-east-1:abc123
    sub = parts[3]           # e.g. cognito sub
    out_key = f"protected/{identity_id}/processed/{sub}/{filename}"

    # Save processed CSV to S3
    out_csv = StringIO()
    processed_df.to_csv(out_csv, index=False)
    s3.put_object(Bucket=bucket, Key=out_key, Body=out_csv.getvalue())

    # 5. Safe value parser
    def safe(val):
        return float(val) if val is not None and np.isfinite(val) else 0.0

    # 6. Build input for mutation
    input_data = {
        "id": session_id,
        "userID": sub,
        "owner": sub,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "durationSeconds": int(len(raw_emg) / 1000),
        "emgS3Key": key,
        "emgProcessedS3Key": out_key,
        "imuS3Key": None,
        "workoutType": "Default",
        "notes": "",
        "peakRMS": safe(metrics.get("peak_magnitude")),
        "averageRMS": safe(metrics.get("avg_magnitude")),
        "fatigueIndex": safe(metrics.get("fatigue_index")),
        "elasticityIndex": safe(metrics.get("elasticity_index")),
        "activationRatio": safe(metrics.get("activation_ratio")),
        "medianFrequency": safe(metrics.get("median_freq")),
        "meanFrequency": safe(metrics.get("mean_freq")),
        "signalToNoiseRatio": safe(metrics.get("snr")),
        "baselineDrift": safe(metrics.get("baseline_drift")),
        "zeroCrossingRate": safe(metrics.get("zcr"))
    }

    # 7. Send GraphQL mutation to AppSync
    mutation = """
    mutation CreateSession($input: CreateSessionInput!) {
      createSession(input: $input) {
        id
      }
    }
    """

    payload = {
        "query": mutation,
        "variables": {
            "input": input_data
        }
    }

    print("GraphQL payload:", json.dumps(payload, indent=2))

    headers = {
        "Content-Type": "application/json",
        "x-api-key": APPSYNC_API_KEY
    }

    response = requests.post(APPSYNC_API_URL, json=payload, headers=headers)
    print("GraphQL response:", response.status_code, response.text)

    return {
        'statusCode': 200,
        'body': json.dumps(f"Processed {key}, saved to {out_key}, metrics sent to AppSync.")
    }
