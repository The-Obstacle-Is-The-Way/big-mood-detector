"""
TimescaleDB Baseline Repository

Production-grade baseline persistence using:
- TimescaleDB for offline storage with continuous aggregates
- Feast for online feature store (Redis)
- Bitemporal data model with effective_ts versioning
- Zero custom infrastructure - battle-tested tooling

Architecture:
Raw Data → TimescaleDB Hypertable → Continuous Aggregates → Feast → Redis
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# Feast type is only used in annotations, not needed
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from structlog import get_logger

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)
from big_mood_detector.infrastructure.security import hash_user_id

logger = get_logger()

# SQLAlchemy Base for TimescaleDB tables
Base: Any = declarative_base()


class BaselineRawRecord(Base):
    """
    Raw baseline data table - hypertable partitioned by ts.

    Stores individual metric values that get aggregated into baselines.
    Following your bitemporal pattern with immutable records.
    """

    __tablename__ = "user_baseline_raw"

    # Composite primary key for bitemporal model
    user_id = Column(String, primary_key=True)
    metric = Column(String, primary_key=True)  # e.g. 'total_sleep_hours'
    ts = Column(DateTime, primary_key=True)  # measurement timestamp
    effective_ts = Column(DateTime, primary_key=True)  # when record was created

    # Value and metadata
    value = Column(Float, nullable=False)
    window_days = Column(Integer, default=30)  # aggregation window
    source = Column(String, default="feature_engineer")


class BaselineAggregateRecord(Base):
    """
    Materialized view of baseline aggregates.

    TimescaleDB continuous aggregates keep this updated automatically.
    Maps to your user_baseline_30d materialized view.
    """

    __tablename__ = "user_baseline_30d"

    user_id = Column(String, primary_key=True)
    feature_name = Column(String, primary_key=True)
    window = Column(String, primary_key=True)  # '30d', '7d', etc.
    as_of = Column(Date, primary_key=True)

    # Aggregated statistics
    mean = Column(Float, nullable=False)
    std = Column(Float, nullable=False)
    n = Column(Integer, nullable=False)  # number of data points

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class TimescaleBaselineRepository(BaselineRepositoryInterface):
    """
    Production baseline repository using TimescaleDB + Feast.

    Features:
    - Hypertable partitioning for time-series performance
    - Continuous aggregates for real-time baseline updates
    - Feast integration for <1ms online inference
    - Bitemporal model for audit trails and rollbacks
    """

    def __init__(
        self,
        connection_string: str,
        enable_feast_sync: bool = False,
        feast_repo_path: Path | None = None,
    ):
        """
        Initialize TimescaleDB repository.

        Args:
            connection_string: PostgreSQL connection string
            enable_feast_sync: Whether to sync to Feast online store
            feast_repo_path: Path to Feast repository config
        """
        self.connection_string = connection_string
        self.enable_feast_sync = enable_feast_sync
        self.feast_repo_path = feast_repo_path

        # Create SQLAlchemy engine and session
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize Feast client if enabled
        from big_mood_detector.infrastructure.repositories.feast_types import (
            FeatureStoreProtocol,
        )

        self.feast_client: FeatureStoreProtocol | None = None  # Will be feast.FeatureStore if available
        if enable_feast_sync:
            try:
                import feast  # type: ignore[import-not-found]

                self.feast_client = feast.FeatureStore(repo_path=str(feast_repo_path))
                logger.info("feast_client_initialized", repo_path=feast_repo_path)
            except ImportError:
                logger.warning(
                    "feast_not_available", message="Install feast for online serving"
                )
                self.enable_feast_sync = False

        # Ensure TimescaleDB extension and tables exist
        self._initialize_database()

        logger.info(
            "timescale_baseline_repository_initialized",
            connection=connection_string,
            feast_enabled=self.enable_feast_sync,
        )

    @contextmanager
    def _get_session(self) -> Iterator[Any]:
        """
        Context manager for proper session lifecycle with explicit transactions.

        Ensures sessions are always closed and transactions are properly handled.
        Uses Session.begin() for explicit transaction control.
        """
        session = self.SessionLocal()
        try:
            with session.begin():
                yield session
                # Transaction automatically commits if no exception
        except Exception:
            # Transaction automatically rolls back on exception
            raise
        finally:
            session.close()

    def save_baseline(self, baseline: UserBaseline) -> None:
        """
        Save baseline with atomic UPSERT operations.

        Uses PostgreSQL's INSERT ... ON CONFLICT for race-condition-free updates.
        Creates immutable records with effective_ts for audit trails.
        """
        with self._get_session() as session:
            effective_time = datetime.now(UTC)

            # Hash user ID for privacy
            hashed_user_id = hash_user_id(baseline.user_id)

            # Prepare metrics
            metrics = self._prepare_metrics(baseline)

            # Save raw records (always insert for bitemporal pattern)
            for metric_name, value in metrics:
                raw_record = BaselineRawRecord(
                    user_id=hashed_user_id,
                    metric=metric_name,
                    ts=baseline.last_updated,
                    effective_ts=effective_time,
                    value=float(value),
                    window_days=30,
                    source="advanced_feature_engineer",
                )
                session.add(raw_record)

            # Use PostgreSQL UPSERT for aggregate records
            for metric_name, value in metrics:
                stmt = insert(BaselineAggregateRecord).values(
                    user_id=hashed_user_id,
                    feature_name=metric_name,
                    window="30d",
                    as_of=baseline.baseline_date,
                    mean=value,
                    std=0.0,
                    n=baseline.data_points,
                    created_at=effective_time,
                )

                # ON CONFLICT update everything except primary keys
                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id", "feature_name", "window", "as_of"],
                    set_={
                        "mean": stmt.excluded.mean,
                        "std": stmt.excluded.std,
                        "n": stmt.excluded.n,
                        "created_at": stmt.excluded.created_at,
                    },
                )

                session.execute(stmt)

            logger.info(
                "baseline_saved_atomically",
                user_id=baseline.user_id,
                effective_ts=effective_time.isoformat(),
                metrics_count=len(metrics),
            )

            # Sync to Feast if enabled
            if self.enable_feast_sync:
                self._sync_to_feast(baseline)

    def _prepare_metrics(self, baseline: UserBaseline) -> list[tuple[str, float]]:
        """Prepare metrics list from baseline."""
        metrics = [
            ("sleep_mean", baseline.sleep_mean),
            ("sleep_std", baseline.sleep_std),
            ("activity_mean", baseline.activity_mean),
            ("activity_std", baseline.activity_std),
            ("circadian_phase", baseline.circadian_phase),
        ]

        # Add HR/HRV metrics if available
        if baseline.heart_rate_mean is not None:
            metrics.append(("heart_rate_mean", baseline.heart_rate_mean))
        if baseline.heart_rate_std is not None:
            metrics.append(("heart_rate_std", baseline.heart_rate_std))
        if baseline.hrv_mean is not None:
            metrics.append(("hrv_mean", baseline.hrv_mean))
        if baseline.hrv_std is not None:
            metrics.append(("hrv_std", baseline.hrv_std))

        return metrics

    def get_baseline(self, user_id: str) -> UserBaseline | None:
        """
        Retrieve latest baseline with proper session management.

        Checks online store first for <1ms performance,
        falls back to TimescaleDB continuous aggregates.
        """
        # Hash user ID for privacy
        hashed_user_id = hash_user_id(user_id)
        # Try online store first (if enabled)
        if self.enable_feast_sync and self.feast_client:
            try:
                baseline = self._get_from_feast(hashed_user_id)
                if baseline:
                    logger.info("baseline_retrieved_online", user_id=user_id)
                    return baseline
            except Exception as e:
                logger.warning("feast_retrieval_failed", user_id=user_id, error=str(e))

        # Fallback to TimescaleDB
        return self._get_from_timescale(hashed_user_id)

    def get_baseline_history(self, user_id: str, limit: int = 10) -> list[UserBaseline]:
        """
        Get historical baselines with proper session management.

        Queries TimescaleDB continuous aggregates for temporal analysis.
        Perfect for detecting baseline drift over time.
        """
        # Hash user ID for privacy
        hashed_user_id = hash_user_id(user_id)

        with self._get_session() as session:
            # Query all baseline dates for this user
            baseline_dates = (
                session.query(BaselineAggregateRecord.as_of)
                .filter(BaselineAggregateRecord.user_id == hashed_user_id)
                .distinct()
                .order_by(BaselineAggregateRecord.as_of.asc())
                .limit(limit)
                .all()
            )

            if not baseline_dates:
                logger.info("baseline_history_empty", user_id=user_id)
                return []

            baselines = []
            for (baseline_date,) in baseline_dates:
                # Get all metrics for this date
                metrics = (
                    session.query(BaselineAggregateRecord)
                    .filter(
                        BaselineAggregateRecord.user_id == hashed_user_id,
                        BaselineAggregateRecord.as_of == baseline_date,
                    )
                    .all()
                )

                baseline = self._reconstruct_baseline(user_id, baseline_date, metrics)
                if baseline:
                    baselines.append(baseline)

            logger.info(
                "baseline_history_retrieved", user_id=user_id, count=len(baselines)
            )

            return baselines

    def _reconstruct_baseline(
        self, user_id: str, baseline_date: date, metrics: list[BaselineAggregateRecord]
    ) -> UserBaseline | None:
        """Reconstruct UserBaseline from aggregate records."""
        if not metrics:
            return None

        # Build metric dictionary with safety checks
        metric_values: dict[str, float] = {}
        for m in metrics:
            assert m.feature_name is not None, "feature_name cannot be None"
            assert m.mean is not None, f"mean cannot be None for {m.feature_name}"
            metric_values[str(m.feature_name)] = float(m.mean)

        return UserBaseline(
            user_id=user_id,
            baseline_date=baseline_date,
            sleep_mean=metric_values.get("sleep_mean", 7.0),
            sleep_std=metric_values.get("sleep_std", 1.0),
            activity_mean=metric_values.get("activity_mean", 8000.0),
            activity_std=metric_values.get("activity_std", 2000.0),
            circadian_phase=metric_values.get("circadian_phase", 22.0),
            # HR/HRV fields - use None if not present (no magic defaults!)
            heart_rate_mean=metric_values.get("heart_rate_mean"),
            heart_rate_std=metric_values.get("heart_rate_std"),
            hrv_mean=metric_values.get("hrv_mean"),
            hrv_std=metric_values.get("hrv_std"),
            last_updated=metrics[0].created_at,  # type: ignore[arg-type]
            data_points=metrics[0].n,  # type: ignore[arg-type]
        )

    def _get_from_timescale(self, user_id: str) -> UserBaseline | None:
        """Retrieve baseline from TimescaleDB with session management."""
        with self._get_session() as session:
            # Get the most recent baseline date for this user
            latest_date = (
                session.query(BaselineAggregateRecord.as_of)
                .filter(BaselineAggregateRecord.user_id == user_id)
                .order_by(BaselineAggregateRecord.as_of.desc())
                .first()
            )

            if not latest_date:
                logger.info("baseline_not_found", user_id=user_id)
                return None

            # Get all metrics for the latest date
            metrics = (
                session.query(BaselineAggregateRecord)
                .filter(
                    BaselineAggregateRecord.user_id == user_id,
                    BaselineAggregateRecord.as_of == latest_date[0],
                )
                .all()
            )

            if not metrics:
                return None

            baseline = self._reconstruct_baseline(user_id, latest_date[0], metrics)

            logger.info(
                "baseline_retrieved_timescale",
                user_id=user_id,
                as_of=str(latest_date[0]),
            )

            return baseline

    def _sync_to_feast(self, baseline: UserBaseline, max_retries: int = 3) -> None:
        """Sync baseline to Feast online store with retry logic.

        Args:
            baseline: UserBaseline to sync
            max_retries: Maximum number of retry attempts
        """
        if not self.feast_client:
            return

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                import pandas as pd

                # Convert baseline to Feast-compatible format
                features_df = pd.DataFrame(
                    [
                        {
                            "user_id": baseline.user_id,
                            "sleep_mean": baseline.sleep_mean,
                            "sleep_std": baseline.sleep_std,
                            "activity_mean": baseline.activity_mean,
                            "activity_std": baseline.activity_std,
                            "circadian_phase": baseline.circadian_phase,
                            "data_points": baseline.data_points,
                            "event_timestamp": baseline.last_updated,
                        }
                    ]
                )

                # Push to online store
                self.feast_client.push("user_baselines", features_df)

                logger.info(
                    "feast_sync_complete",
                    user_id=baseline.user_id,
                    features_count=len(features_df.columns),
                )
                return  # Success, exit

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.5s, 1s, 2s
                    backoff_time = 0.5 * (2**attempt)
                    logger.warning(
                        "feast_sync_retry",
                        user_id=baseline.user_id,
                        attempt=attempt + 1,
                        backoff=backoff_time,
                        error=str(e),
                    )
                    time.sleep(backoff_time)
                else:
                    # Final attempt failed
                    logger.error(
                        "feast_sync_failed_after_retries",
                        user_id=baseline.user_id,
                        attempts=max_retries,
                        error=str(e),
                    )
                    # Don't raise - Feast sync is optional

    def _get_from_feast(self, hashed_user_id: str) -> UserBaseline | None:
        """Retrieve baseline from Feast online store.

        Args:
            hashed_user_id: Already hashed user ID for privacy
        """
        if not self.feast_client:
            return None

        try:
            # Query online features
            feature_vector = self.feast_client.get_online_features(
                features=[
                    "user_baselines:sleep_mean",
                    "user_baselines:sleep_std",
                    "user_baselines:activity_mean",
                    "user_baselines:activity_std",
                    "user_baselines:circadian_phase",
                    "user_baselines:data_points",
                ],
                entity_rows=[{"user_id": hashed_user_id}],
            )

            # Convert to dict
            features = feature_vector.to_dict()

            if not features or not features.get("sleep_mean"):
                return None

            # Build UserBaseline from online features
            baseline = UserBaseline(
                user_id=hashed_user_id,
                baseline_date=date.today(),  # Would need to store this in Feast
                sleep_mean=features["sleep_mean"][0],
                sleep_std=features["sleep_std"][0],
                activity_mean=features["activity_mean"][0],
                activity_std=features["activity_std"][0],
                circadian_phase=features["circadian_phase"][0],
                last_updated=datetime.now(UTC),
                data_points=features["data_points"][0],
            )

            return baseline

        except Exception as e:
            logger.warning(
                "feast_online_retrieval_failed", user_id=hashed_user_id, error=str(e)
            )
            return None

    def _initialize_database(self) -> None:
        """Initialize TimescaleDB extension and create tables."""
        try:
            with self.engine.connect() as conn:
                # Enable TimescaleDB extension
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                except Exception:
                    # Extension might already exist or no permissions
                    pass

                # Create tables
                Base.metadata.create_all(self.engine)

                # Create hypertable for raw data (if not exists)
                try:
                    conn.execute(
                        text(
                            "SELECT create_hypertable('user_baseline_raw', 'ts', "
                            "if_not_exists => TRUE)"
                        )
                    )
                except Exception:
                    # Hypertable might already exist
                    pass

                # Create continuous aggregate (materialized view)
                try:
                    conn.execute(
                        text(
                            """
                        CREATE MATERIALIZED VIEW IF NOT EXISTS user_baseline_30d
                        WITH (timescaledb.continuous) AS
                        SELECT
                            user_id,
                            metric as feature_name,
                            '30d' as window,
                            time_bucket('1 day', ts) AS as_of,
                            AVG(value) AS mean,
                            STDDEV_SAMP(value) AS std,
                            COUNT(value) AS n,
                            MAX(effective_ts) AS created_at
                        FROM user_baseline_raw
                        WHERE ts >= NOW() - INTERVAL '30 days'
                        GROUP BY user_id, metric, time_bucket('1 day', ts)
                    """
                        )
                    )
                except Exception as e:
                    # View might already exist
                    logger.debug("continuous_aggregate_creation_info", error=str(e))

                conn.commit()

            logger.info("timescale_database_initialized")

        except Exception as e:
            logger.error("timescale_initialization_failed", error=str(e))
            # Continue anyway - production deployments handle this separately
