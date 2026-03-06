#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

class AnomalyDetector:

    def __init__(self, z_threshold: float = 3.0, contamination: float = 0.05):
        if not isinstance(z_threshold, (int, float)) or z_threshold <= 0:
            raise ValueError(f"z_threshold must be a positive number, got {z_threshold!r}")
        if not isinstance(contamination, float) or not (0 < contamination < 0.5):
            raise ValueError(f"contamination must be a float in (0, 0.5), got {contamination!r}")

        self.z_threshold = z_threshold
        self.contamination = contamination  # expected fraction of anomalies
        logger.info(f"AnomalyDetector initialized with z_threshold={z_threshold}, contamination={contamination}")

    def zscore_flag(
        self,
        values: pd.Series,
        mean: float,
        std: float
    ) -> pd.Series:
        """
        Flag values more than z_threshold standard deviations from the
        established baseline mean. Returns a Series of z-scores.
        """
        if not isinstance(values, pd.Series):
            raise ValueError("values must be a pandas Series.")
        if not isinstance(mean, (int, float)):
            raise ValueError("mean must be a number.")
        if not isinstance(std, (int, float)) or std < 0:
            raise ValueError("std must be a non-negative number.")
        
        if std == 0:
            logger.debug("Standard deviation is zero, cannot compute z-scores. Returning zeros.")
            return pd.Series([0.0] * len(values), index=values.index)
        try:
            z_scores = (values - mean).abs() / std
        except Exception as e:
            logger.error(f"Error computing z-scores: {e}")
            raise

        logger.debug(f"Computed z-scores for {len(values)} values with mean={mean} and std={std}.")
        return z_scores

    def isolation_forest_flag(self, df: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
        """
        Multivariate anomaly detection across all numeric channels simultaneously.
        IsolationForest returns -1 for anomalies, 1 for normal points.
        Scores closer to -1 indicate stronger anomalies.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not numeric_cols:
            raise ValueError("numeric_cols must be a non-empty list")
        
        missing = [col for col in numeric_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns missing from DataFrame: {missing}")
        if len(df) < 2:
            raise ValueError("DataFrame must have at least 2 rows for IsolationForest to work.")
        
        logger.info(f"Running IsolationForest on {len(df)} rows and {len(numeric_cols)} numeric columns.")

        try: 
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            model.fit(X)

            labels = model.predict(X)          # -1 = anomaly, 1 = normal
            scores = model.decision_function(X)  # lower = more anomalous
        except Exception as e:
            logger.error(f"Error running IsolationForest: {e}")
            raise 

        n_anomalies = (labels == -1).sum()
        logger.info(f"IsolationForest detected {n_anomalies} anomalies out of {len(df)} rows.")
        return labels, scores

    def run(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        baseline: dict,
        method: str = "both"
    ) -> pd.DataFrame:
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if not numeric_cols:
            raise ValueError("numeric_cols must be a non-empty list.")
        if method not in ("zscore", "isolation", "both"):
            raise ValueError("method must be one of 'zscore', 'isolation', or 'both'.")
        
        missing = [col for col in numeric_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns missing from DataFrame: {missing}")
        
        logger.info(f"Running anomaly detection with method='{method}' on DataFrame with {len(df)} rows and {len(numeric_cols)} numeric columns.")

        result = df.copy()

        # --- Z-score per channel ---
        if method in ("zscore", "both"):
            for col in numeric_cols:
                stats = baseline.get(col)
                if stats and stats["count"] >= 30:  # need enough history to trust baseline
                    try: 
                        z_scores = self.zscore_flag(df[col], stats["mean"], stats["std"])
                        result[f"{col}_zscore"] = z_scores.round(4)
                        result[f"{col}_zscore_flag"] = z_scores > self.z_threshold
                        logger.debug(f"Z-score flags computed for column '{col}' with {result[f'{col}_zscore_flag'].sum()} anomalies flagged.")
                    except Exception as e:
                        logger.error(f"Error processing column '{col}' for z-score: {e}")
                        result[f"{col}_zscore"] = None
                        result[f"{col}_zscore_flag"] = None
                else:
                    # Not enough baseline history yet — flag as unknown
                    count = stats.get("count", 0) if stats else 0
                    logger.warning("Insufficient baseline history for channel '%s' | count=%d < 30 — skipping z-score", col, count)
                    result[f"{col}_zscore"] = None
                    result[f"{col}_zscore_flag"] = None

        # --- IsolationForest across all channels ---
        if method in ("isolation", "both"):
            try: 
                labels, scores = self.isolation_forest_flag(df, numeric_cols)
                result["if_label"] = labels          # -1 or 1
                result["if_score"] = scores.round(4) # continuous anomaly score
                result["if_flag"] = labels == -1
            except Exception as e:
                logger.error(f"Error running IsolationForest: {e}")
                raise

        # --- Consensus flag: anomalous by at least one method ---
        if method == "both":
            zscore_flags = [
                result[f"{col}_zscore_flag"]
                for col in numeric_cols
                if f"{col}_zscore_flag" in result.columns
                and result[f"{col}_zscore_flag"].notna().any()
            ]
            if zscore_flags:
                any_zscore = pd.concat(zscore_flags, axis=1).any(axis=1)
                result["anomaly"] = any_zscore | result["if_flag"]
            else:
                logger.warning("No valid z-score flags available to combine with IsolationForest results.")
                result["anomaly"] = result["if_flag"]
            
            total = int(result["anomaly"].sum())
            logger.info(f"Total anomalies flagged by combined method: {total} out of {len(result)} rows.")

        return result
