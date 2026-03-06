#!/usr/bin/env python3
import json
import math
import boto3
from datetime import datetime
from typing import Optional
import logging
from botocore.exceptions import ClientError

s3 = boto3.client("s3")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BaselineManager:

    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        if not bucket:
            raise ValueError("Bucket name must be provided for BaselineManager.")
        if not baseline_key:
            raise ValueError("Baseline key must be provided for BaselineManager.")
        self.bucket = bucket
        self.baseline_key = baseline_key
        logger.info(f"BaselineManager initialized with bucket: {bucket}, key: {baseline_key}")

    def load(self) -> dict:
        logger.info("Loading baseline from S3.")
        try:
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            logger.info("Baseline loaded successfully.")
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning("Baseline file not found. Starting with empty baseline.")
                return {}
            else:
                logger.error(f"Failed to load baseline: {e}")
                raise

    def save(self, baseline: dict):
        logger.info("Saving baseline.")
        if not isinstance(baseline, dict):
            raise ValueError("Baseline must be a dictionary.")
        baseline["last_updated"] = datetime.utcnow().isoformat()

        try:
            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )
            logger.info("Baseline saved to S3 | key=%s", self.baseline_key)
        except Exception as e:
            logger.error("Failed to save baseline | error=%s", e)
            raise

        log_path = "/var/log/anomaly-app.log"
        try:
            with open(log_path, "rb") as f:
                s3.put_object(
                    Bucket=self.bucket,
                    Key="logs/anomaly-app.log",
                    Body=f.read(),
                    ContentType="text/plain"
                )
            logger.info("Log file synced to S3 | key=logs/anomaly-app.log")
        except Exception as e:
            logger.error("Failed to sync log file to S3 | error=%s", e)

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        if channel not in baseline:
            baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}
        state = baseline[channel]
        try:
            new_values = [float(v) for v in new_values if v is not None]
            if not all(math.isfinite(v) for v in new_values):
                raise ValueError("Non-finite value encountered in new_values.")
        except ValueError as e:
            logger.error(f"Non-numeric value encountered in new_values for channel '{channel}': {e}")
            raise
        for value in new_values:
            state["count"] += 1
            delta = value - state["mean"]
            state["mean"] += delta / state["count"]
            delta2 = value - state["mean"]
            state["M2"] += delta * delta2
        if state["count"] >= 2:
            variance = state["M2"] / state["count"]
            state["std"] = math.sqrt(variance)
        else:
            state["std"] = 0.0
        baseline[channel] = state
        logger.info(f"Updated baseline for channel '{channel}': count={state['count']}, mean={state['mean']:.4f}, std={state['std']:.4f}")
        return baseline

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        if channel not in baseline:
            logger.warning(f"Channel '{channel}' not found in baseline.")
            return None
        return baseline.get(channel)
