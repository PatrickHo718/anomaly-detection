# app.py
import io
import json
import os
import boto3
import pandas as pd
import requests
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from baseline import BaselineManager
from processor import process_file
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%dT%H:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anomaly Detection Pipeline")

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]
if not BUCKET_NAME:
    logger.error("BUCKET_NAME environment variable is not set.")
    raise ValueError("BUCKET_NAME environment variable must be set.")
logger.info(f"App initialized with bucket: {BUCKET_NAME}")

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    try: 
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON body from SNS message: {e}")
        raise ValueError("Invalid JSON body in request.") from e
    msg_type = request.headers.get("x-amz-sns-message-type")
    logger.info(f"Received SNS message | type={msg_type} | body_keys={list(body.keys())}")

    # SNS sends a SubscriptionConfirmation before it will deliver any messages.
    # Visiting the SubscribeURL confirms the subscription.
    if msg_type == "SubscriptionConfirmation":
        confirm_url = body["SubscribeURL"]
        if not confirm_url:
            logger.error("SubscriptionConfirmation message missing SubscribeURL.")
            raise ValueError("Missing SubscribeURL in SubscriptionConfirmation message.")
        try:
            resp = requests.get(confirm_url, timeout=10)
            resp.raise_for_status()
            logger.info("SNS subscription confirmed | url=%s", confirm_url)
        except requests.RequestException as e:
            logger.error("Failed to confirm SNS subscription | url=%s | error=%s", confirm_url, e)
            raise RuntimeError("Failed to confirm SNS subscription.") from e
        return {"status": "confirmed"}

    if msg_type == "Notification":
        # The SNS message body contains the S3 event as a JSON string
        raw_message = body.get("Message")
        if not raw_message:
            logger.error("Notification message missing 'Message' field.")
            raise ValueError("Missing 'Message' field in Notification message.")
        try: 
            s3_event = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse S3 event from Notification message: {e}")
            raise ValueError("Invalid JSON in Notification message.") from e
        
        records = s3_event.get("Records", [])
        logger.info("S3 event received | records=%d", len(records))

        for record in records:
            try: 
                key = record["s3"]["object"]["key"]
            except KeyError as e:
                logger.error(f"Malformed S3 event record: missing key {e}")
                continue

            if key.startswith("raw/") and key.endswith(".csv"):
                logger.info(f"Scheduling processing for new file: s3://{BUCKET_NAME}/{key}")
                background_tasks.add_task(process_file, BUCKET_NAME, key)
            else:
                logger.info(f"Ignoring S3 event for key: {key} (not in 'raw/' or not a .csv)") 

    return {"status": "ok"}


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000.")
    
    try: 
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        keys = sorted(
            [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ],
            reverse=True,
        )[:10]
    except Exception as e:
        logger.error(f"Error listing processed files from S3: {e}")
        raise RuntimeError("Failed to list processed files from S3.") from e

    if not keys:
        logger.info("No processed files found in S3.")
        return {"count": 0, "anomalies": []}
    logger.info(f"Found {len(keys)} processed files to scan for anomalies.")

    all_anomalies = []
    for key in keys:
        try: 
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            if "anomaly" in df.columns:
                flagged = df[df["anomaly"] == True].copy()
                flagged["source_file"] = key
                all_anomalies.append(flagged)
        except Exception as e:
            logger.error(f"Error processing file {key} for recent anomalies: {e}")
            continue

    if not all_anomalies:
        return {"count": 0, "anomalies": []}
    
    combined = pd.concat(all_anomalies).head(limit)
    logger.info(f"Returning {len(combined)} recent anomalies across {len(keys)} files.")
    return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}


@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    try: 
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")
    except Exception as e:
        logger.error(f"Error listing processed files from S3: {e}")
        raise RuntimeError("Failed to list processed files from S3.") from e

    summaries = []
    for page in pages:
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("_summary.json"):
                continue
            try: 
                response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                summaries.append(json.loads(response["Body"].read()))
            except Exception as e:
                logger.error(f"Error loading summary file {obj['Key']}: {e}")
                continue

    if not summaries:
        return {"message": "No processed files yet."}

    total_rows = sum(s["total_rows"] for s in summaries)
    total_anomalies = sum(s["anomaly_count"] for s in summaries)

    return {
        "files_processed": len(summaries),
        "total_rows_scored": total_rows,
        "total_anomalies": total_anomalies,
        "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
        "most_recent": sorted(summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
    }


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    try: 
        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        baseline = baseline_mgr.load()
    except Exception as e:
        logger.error(f"Error loading baseline: {e}")
        raise RuntimeError("Failed to load baseline from S3.") from e

    channels = {}
    for channel, stats in baseline.items():
        if channel == "last_updated":
            continue
        if not isinstance(stats, dict):
            logger.warning(f"Skipping malformed baseline entry for channel '{channel}': expected dict, got {type(stats)}")
            continue
        channels[channel] = {
            "observations": stats["count"],
            "mean": round(stats["mean"], 4),
            "std": round(stats.get("std", 0.0), 4),
            "baseline_mature": stats["count"] >= 30,
        }

    logger.info(f"Current baseline retrieved with {len(channels)} channels.")
    return {
        "last_updated": baseline.get("last_updated"),
        "channels": channels,
    }


@app.get("/health")
def health():
    return {"status": "ok", "bucket": BUCKET_NAME, "timestamp": datetime.utcnow().isoformat()}
