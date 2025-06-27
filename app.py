import json
import boto3
import os
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import requests
from urllib.parse import unquote_plus
from emg_processor import process_emg_dataframe

s3 = boto3.client('s3')

APPSYNC_API_URL = os.environ.get("APPSYNC_API_URL")
APPSYNC_API_KEY = os.environ.get("APPSYNC_API_KEY")

def lambda_handler(event, context):
    record = event['Records'][0]['s3']
    bucket = record['bucket']['name']
    key = unquote_plus(record['object']['key'])
    print(f"Triggered on key: {key}")

    if '/emg/' not in key or '/processed/' in key or not key.endswith('.csv'):
        print("Skipping file not in /emg/ or already processed.")
        return {'statusCode': 200, 'body': 'Skipped'}

    filename = key.split("/")[-1]
    session_id = filename.replace(".csv", "")
    parts = key.split('/')
    identity_id = parts[1]
    sub = parts[3]

    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    df.columns = df.columns.str.strip()

    has_time_col = 'time' in df.columns
    all_metrics = {}
    output_df = pd.DataFrame()

    emg_columns = [col for col in df.columns if col.lower().startswith("emg")]

    for col in emg_columns:
        raw = pd.to_numeric(df[col], errors='coerce')
        input_df = pd.DataFrame()
        if has_time_col:
            input_df['time'] = df['time']
        input_df[col] = raw

        proc_df, metrics = process_emg_dataframe(input_df)
        prefix = col

        if 'time' in proc_df.columns and 'time' not in output_df.columns:
            output_df['time'] = proc_df['time']

        output_df[f"{prefix}_filtered"] = proc_df[f"{col}_filtered"]
        output_df[f"{prefix}_rms"] = proc_df[f"{col}_rms"]

        for k, v in metrics.items():
            all_metrics[k] = float(v) if np.isfinite(v) else 0.0

    out_key = f"protected/{identity_id}/processed/{sub}/{filename}"
    out_csv = StringIO()
    output_df.to_csv(out_csv, index=False)
    s3.put_object(Bucket=bucket, Key=out_key, Body=out_csv.getvalue())

    def safe(val):
        return float(val) if val is not None and np.isfinite(val) else 0.0

    input_data = {
        "id": session_id,
        "userID": sub,
        "owner": sub,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "durationSeconds": int(len(df) / 1000),
        "emgS3Key": key,
        "emgProcessedS3Key": out_key,
        "imuS3Key": None,
        "workoutType": "Default",
        "notes": ""
    }

    gql_suffixes = {
        "peak_magnitude": "peakRMS",
        "avg_magnitude": "averageRMS",
        "fatigue_index": "fatigueIndex",
        "elasticity_index": "elasticityIndex",
        "activation_ratio": "activationRatio",
        "median_freq": "medianFrequency",
        "mean_freq": "meanFrequency",
        "signalToNoiseRatio": "signalToNoiseRatio",
        "baseline_drift": "baselineDrift",
        "zcr": "zeroCrossingRate",
        "rate_of_rise": "rateOfRise",
        "rate_of_fall": "rateOfFall",
        "rfd_analog": "rfdAnalog",
        "time_snr_raw": "snrTimeRaw",
        "time_snr_denoised": "snrTimeDenoised",
        "freq_snr_raw": "snrFreqRaw",
        "freq_snr_denoised": "snrFreqDenoised"
    }

    valid_fields = {
        "emg_ch1_peakRMS", "emg_ch1_averageRMS", "emg_ch1_fatigueIndex", "emg_ch1_elasticityIndex",
        "emg_ch1_activationRatio", "emg_ch1_medianFrequency", "emg_ch1_meanFrequency",
        "emg_ch1_signalToNoiseRatio", "emg_ch1_baselineDrift", "emg_ch1_zeroCrossingRate",
        "emg_ch1_rateOfRise", "emg_ch1_rateOfFall", "emg_ch1_rfdAnalog", "emg_ch1_snrTimeRaw",
        "emg_ch1_snrTimeDenoised", "emg_ch1_snrFreqRaw", "emg_ch1_snrFreqDenoised",

        "emg_ch2_peakRMS", "emg_ch2_averageRMS", "emg_ch2_fatigueIndex", "emg_ch2_elasticityIndex",
        "emg_ch2_activationRatio", "emg_ch2_medianFrequency", "emg_ch2_meanFrequency",
        "emg_ch2_signalToNoiseRatio", "emg_ch2_baselineDrift", "emg_ch2_zeroCrossingRate",
        "emg_ch2_rateOfRise", "emg_ch2_rateOfFall", "emg_ch2_rfdAnalog", "emg_ch2_snrTimeRaw",
        "emg_ch2_snrTimeDenoised", "emg_ch2_snrFreqRaw", "emg_ch2_snrFreqDenoised"
    }

    for k, v in all_metrics.items():
        for raw_suffix, gql_suffix in gql_suffixes.items():
            if k.endswith(raw_suffix):
                gql_key = k.replace(raw_suffix, gql_suffix)
                print(f"Trying: {gql_key}")
                if gql_key in valid_fields:
                    input_data[gql_key] = safe(v)
                    print(f"Added: {gql_key}")
                else:
                    print(f"Skipped: {gql_key} not in schema")
                break

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
