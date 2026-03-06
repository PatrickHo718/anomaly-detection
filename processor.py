#!/usr/bin/env python3
import json
import io
import boto3
import pandas as pd
from datetime import datetime
import logging
from baseline import BaselineManager
from detector import AnomalyDetector

s3 = boto3.client("s3")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]  # students configure this

def process_file(bucket: str, key: str):
    if not bucket or not key:
        raise ValueError("Both bucket and key must be provided to process_file.")
    logger.info(f"Processing file: s3://{bucket}/{key}")

    # 1. Download raw file
    try: 
         response = s3.get_object(Bucket=bucket, Key=key)
         df = pd.read_csv(io.BytesIO(response["Body"].read()))
         logger.info(f"Loaded file with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
         logger.error(f"Error loading file from S3: {e}")
         raise

    # 2. Load current baseline
    try: 
        baseline_mgr = BaselineManager(bucket=bucket)
        baseline = baseline_mgr.load()
    except Exception as e:
        logger.error(f"Error loading baseline: {e}")
        raise

    # 3. Update baseline with values from this batch BEFORE scoring
    #    (use only non-null values for each channel)
    for col in NUMERIC_COLS:
        if col not in df.columns:
            logger.debug("Skipping baseline update — column not in file | col=%s", col)
            continue
        clean_values = df[col].dropna().tolist()
        if not clean_values:
            logger.warning("All values are null for column — skipping baseline update | col=%s", col)
            continue
        try:
            baseline = baseline_mgr.update(baseline, col, clean_values)
            logger.debug("Baseline updated | col=%s | n_values=%d", col, len(clean_values))
        except Exception as e:
            logger.error("Baseline update failed for column — skipping | col=%s | error=%s", col, e)

    # 4. Run detection
    active_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if not active_cols:
        raise ValueError("None of the expected numeric columns are present in the file for detection.")

    try:
        detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
        scored_df = detector.run(df, active_cols, baseline, method="both")
    except Exception as e:
        logger.error("Anomaly detection failed | error=%s", e)
        raise 

    logger.info("Detection complete | rows=%d", len(scored_df))

    # 5. Write scored file to processed/ prefix
    output_key = key.replace("raw/", "processed/")
    try: 
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )
        logger.info(f"Scored file written to s3://{bucket}/{output_key}")
    except Exception as e:
        logger.error(f"Error writing scored file to S3: {e}")
    
    # 6. Save updated baseline back to S3
    try:
        baseline_mgr.save(baseline)
        logger.info("Updated baseline saved to S3.")
    except Exception as e:
        logger.error(f"Error saving updated baseline to S3: {e}")

    # 7. Build and return a processing summary
    anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df.columns else 0
    summary = {
        "source_key": key,
        "output_key": output_key,
        "processed_at": datetime.utcnow().isoformat(),
        "total_rows": len(df),
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
        "baseline_observation_counts": {
            col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
        }
    }

    # Write summary JSON alongside the processed file
    summary_key = output_key.replace(".csv", "_summary.json")
    try: 
        s3.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json"
        )
        logger.info(f"Summary JSON written to s3://{bucket}/{summary_key}")
    except Exception as e:
        logger.error(f"Error writing summary JSON to S3: {e}")

    logger.info(
        "Processing complete | anomalies=%d / %d (%.1f%%) | source=%s",
        anomaly_count, len(df), 100 * anomaly_count / len(df) if len(df) > 0 else 0, key
    )
    return summary
