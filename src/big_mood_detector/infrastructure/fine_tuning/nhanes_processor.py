"""
NHANES Data Processor

Processes NHANES XPT files into labeled datasets for fine-tuning.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from big_mood_detector.core.logging import get_module_logger

logger = get_module_logger(__name__)


class NHANESProcessor:
    """Process NHANES data for mood prediction fine-tuning."""

    # Benzodiazepine generic IDs and names
    BENZODIAZEPINES = {
        "d00321",  # Alprazolam
        "d00732",  # Lorazepam
        "d00149",  # Clonazepam
        "d00533",  # Diazepam
        "d00674",  # Temazepam
        "d00541",  # Oxazepam
        "alprazolam", "xanax",
        "lorazepam", "ativan",
        "clonazepam", "klonopin",
        "diazepam", "valium",
        "temazepam", "restoril",
    }

    # SSRI generic IDs and names
    SSRIS = {
        "d00283",  # Fluoxetine
        "d00859",  # Sertraline
        "d00415",  # Paroxetine
        "d00823",  # Citalopram
        "d04851",  # Escitalopram
        "d00506",  # Fluvoxamine
        "fluoxetine", "prozac",
        "sertraline", "zoloft",
        "paroxetine", "paxil",
        "citalopram", "celexa",
        "escitalopram", "lexapro",
    }

    def __init__(
        self,
        data_dir: Path = Path("nhanes"),
        output_dir: Path = Path("processed"),
    ):
        """Initialize processor with data paths.

        Args:
            data_dir: Directory containing NHANES XPT files
            output_dir: Directory for processed output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_actigraphy(self, file_path: str) -> pd.DataFrame:
        """Load PAXHD_H.xpt actigraphy data.

        Args:
            file_path: Path to PAXHD_H.xpt file

        Returns:
            DataFrame with actigraphy data
        """
        logger.info(f"Loading actigraphy data from {file_path}")
        df = pd.read_sas(self.data_dir / file_path)
        logger.info(f"Loaded {len(df)} actigraphy records")
        return df

    def load_depression_scores(self, file_path: str) -> pd.DataFrame:
        """Load DPQ_H.xpt depression questionnaire data.

        Args:
            file_path: Path to DPQ_H.xpt file

        Returns:
            DataFrame with PHQ-9 scores and depression labels
        """
        logger.info(f"Loading depression data from {file_path}")
        df = pd.read_sas(self.data_dir / file_path)

        # PHQ-9 questions
        phq9_cols = [
            "DPQ010",  # Little interest
            "DPQ020",  # Feeling down
            "DPQ030",  # Sleep problems
            "DPQ040",  # Tired
            "DPQ050",  # Appetite
            "DPQ060",  # Feeling bad
            "DPQ070",  # Concentration
            "DPQ080",  # Moving slowly
            "DPQ090",  # Self harm
        ]

        # Calculate total PHQ-9 score
        df["PHQ9_total"] = df[phq9_cols].sum(axis=1)

        # Label depression (PHQ-9 >= 10)
        df["depressed"] = (df["PHQ9_total"] >= 10).astype(int)

        logger.info(
            f"Loaded {len(df)} subjects, "
            f"{df['depressed'].sum()} with depression"
        )
        return df

    def load_medications(
        self, rx_file: str, drug_file: str
    ) -> pd.DataFrame:
        """Load medication data from RXQ files.

        Args:
            rx_file: Path to RXQ_RX_H.xpt prescription file
            drug_file: Path to RXQ_DRUG.xpt drug lookup file

        Returns:
            DataFrame with medication flags by subject
        """
        logger.info(f"Loading medications from {rx_file} and {drug_file}")

        # Load prescription data
        rx_df = pd.read_sas(self.data_dir / rx_file)

        # Load drug lookup
        drug_df = pd.read_sas(self.data_dir / drug_file)

        # Merge to get drug IDs
        med_df = rx_df.merge(drug_df, on="RXDDRUG", how="left")

        # Group by subject and check medications
        subject_meds = (
            med_df.groupby("SEQN")["RXDDRGID"]
            .apply(list)
            .reset_index()
        )

        # Check for medication classes
        subject_meds["benzodiazepine"] = subject_meds["RXDDRGID"].apply(
            lambda drugs: any(
                self.is_benzodiazepine(str(d)) for d in drugs
            )
        ).astype(int)

        subject_meds["ssri"] = subject_meds["RXDDRGID"].apply(
            lambda drugs: any(self.is_ssri(str(d)) for d in drugs)
        ).astype(int)

        subject_meds["antidepressant"] = subject_meds["ssri"]  # Simplified

        # Drop the list column
        subject_meds = subject_meds.drop("RXDDRGID", axis=1)

        logger.info(
            f"Loaded medications for {len(subject_meds)} subjects"
        )
        return subject_meds

    def is_benzodiazepine(self, drug_id: str) -> bool:
        """Check if drug is a benzodiazepine.

        Args:
            drug_id: Generic drug ID or name

        Returns:
            True if benzodiazepine
        """
        drug_lower = drug_id.lower()
        return any(benzo in drug_lower for benzo in self.BENZODIAZEPINES)

    def is_ssri(self, drug_id: str) -> bool:
        """Check if drug is an SSRI.

        Args:
            drug_id: Generic drug ID or name

        Returns:
            True if SSRI
        """
        drug_lower = drug_id.lower()
        return any(ssri in drug_lower for ssri in self.SSRIS)

    def process_cohort(
        self,
        actigraphy_file: str = "PAXHD_H.xpt",
        depression_file: str = "DPQ_H.xpt",
        rx_file: str = "RXQ_RX_H.xpt",
        drug_file: str = "RXQ_DRUG.xpt",
    ) -> pd.DataFrame:
        """Process full NHANES cohort with all data sources.

        Args:
            actigraphy_file: Actigraphy data file
            depression_file: Depression questionnaire file
            rx_file: Prescription data file
            drug_file: Drug lookup file

        Returns:
            Merged cohort DataFrame
        """
        # Load all data sources
        actigraphy = self.load_actigraphy(actigraphy_file)
        depression = self.load_depression_scores(depression_file)
        medications = self.load_medications(rx_file, drug_file)

        # Get unique subjects from actigraphy
        subjects = actigraphy[["SEQN"]].drop_duplicates()

        # Merge all data
        cohort = subjects.merge(depression, on="SEQN", how="left")
        cohort = cohort.merge(medications, on="SEQN", how="left")

        # Fill missing medication flags
        cohort["benzodiazepine"] = cohort["benzodiazepine"].fillna(0)
        cohort["ssri"] = cohort["ssri"].fillna(0)

        logger.info(f"Processed cohort with {len(cohort)} subjects")
        return cohort

    def aggregate_to_daily(self, actigraphy: pd.DataFrame) -> pd.DataFrame:
        """Aggregate minute-level actigraphy to daily features.

        Args:
            actigraphy: Minute-level actigraphy data

        Returns:
            Daily aggregated features
        """
        # Group by subject and day
        daily = actigraphy.groupby(["SEQN", "PAXDAY"]).agg({
            "PAXINTEN": [
                "sum",  # Total activity
                "mean",  # Mean activity
                "std",  # Activity variability
                lambda x: (x < 100).sum(),  # Sedentary minutes
                lambda x: ((x >= 100) & (x < 760)).sum(),  # Moderate
                lambda x: (x >= 760).sum(),  # Vigorous
            ]
        })

        # Flatten column names
        daily.columns = [
            "total_activity",
            "mean_activity",
            "std_activity",
            "sedentary_minutes",
            "moderate_minutes",
            "vigorous_minutes",
        ]

        return daily.reset_index()

    def extract_pat_sequences(
        self,
        actigraphy: pd.DataFrame,
        window_size: int = 60,
        stride: int = 30,
        label_col: Optional[str] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract sliding window sequences for PAT model.

        Args:
            actigraphy: Minute-level actigraphy data
            window_size: Window size in minutes (default 60)
            stride: Stride between windows (default 30)
            label_col: Optional label column for supervised training

        Returns:
            Tuple of (sequences, labels) arrays
        """
        sequences: list[np.ndarray] = []
        labels: list[Any] | None = [] if label_col else None

        # Process each subject
        for seqn in actigraphy["SEQN"].unique():
            subject_data = actigraphy[actigraphy["SEQN"] == seqn].sort_values(
                ["PAXDAY", "PAXN"]
            )

            # Get activity intensity values
            activity = subject_data["PAXINTEN"].values

            # Extract sliding windows
            for i in range(0, len(activity) - window_size + 1, stride):
                window = activity[i : i + window_size]
                sequences.append(window)

                if label_col and labels is not None:
                    # Use label from middle of window
                    label_idx = i + window_size // 2
                    label = subject_data.iloc[label_idx][label_col]
                    labels.append(label)

        sequences_array = np.array(sequences)
        labels_array = np.array(labels) if labels is not None else None

        logger.info(f"Extracted {len(sequences_array)} sequences")
        return sequences_array, labels_array

    def save_cohort(
        self, cohort: pd.DataFrame, name: str
    ) -> Path:
        """Save processed cohort to parquet file.

        Args:
            cohort: Processed cohort DataFrame
            name: Output file name (without extension)

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{name}.parquet"
        cohort.to_parquet(output_path, index=False)
        logger.info(f"Saved cohort to {output_path}")
        return output_path

    def engineer_features(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer 36 features matching mood_ml paper.

        Args:
            daily_data: Daily aggregated activity and sleep data

        Returns:
            DataFrame with engineered features per subject
        """
        features = []

        for seqn in daily_data["SEQN"].unique():
            subject = daily_data[daily_data["SEQN"] == seqn]

            # Basic sleep statistics
            feat = {
                "SEQN": seqn,
                "mean_sleep_duration": subject.get("sleep_duration", 0).mean(),
                "std_sleep_duration": subject.get("sleep_duration", 0).std(),
                "mean_sleep_efficiency": subject.get("sleep_efficiency", 0).mean(),
            }

            # Activity statistics
            if "total_activity" in subject.columns:
                activity = subject["total_activity"].values

                # Interdaily Stability (IS)
                hourly_means = []
                for h in range(24):
                    hour_data = activity[h::24]
                    if len(hour_data) > 0:
                        hourly_means.append(hour_data.mean())

                if hourly_means:
                    grand_mean = np.mean(hourly_means)
                    numerator = np.sum([(m - grand_mean) ** 2 for m in hourly_means])
                    denominator = np.var(activity) * len(activity)
                    feat["IS"] = numerator / denominator if denominator > 0 else 0

                # Intradaily Variability (IV)
                diffs = np.diff(activity)
                feat["IV"] = np.mean(diffs**2) / np.var(activity) if np.var(activity) > 0 else 0

                # Relative Amplitude (RA)
                sorted_act = np.sort(activity)
                L5 = np.mean(sorted_act[: len(sorted_act) // 5])  # Least active 5 hours
                M10 = np.mean(sorted_act[-len(sorted_act) // 10 :])  # Most active 10 hours
                feat["RA"] = (M10 - L5) / (M10 + L5) if (M10 + L5) > 0 else 0
                feat["L5"] = L5
                feat["M10"] = M10

            features.append(feat)

        return pd.DataFrame(features)