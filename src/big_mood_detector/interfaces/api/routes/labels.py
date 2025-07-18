"""
Label Management API Routes

CRUD operations for mood episode labels and baseline periods.
"""

from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
    SQLiteEpisodeRepository,
)

router = APIRouter(prefix="/api/v1/labels", tags=["labels"])


class EpisodeCreateRequest(BaseModel):
    """Request to create a new episode label."""

    start_date: date
    end_date: date | None = None
    episode_type: str = Field(..., pattern="^(depressive|hypomanic|manic|mixed)$")
    severity: int = Field(..., ge=1, le=10)
    notes: str = ""
    rater_id: str = "api_user"


class BaselineCreateRequest(BaseModel):
    """Request to create a baseline period."""

    start_date: date
    end_date: date
    notes: str = ""
    rater_id: str = "api_user"


class EpisodeResponse(BaseModel):
    """Episode response model."""

    id: str
    start_date: date
    end_date: date
    episode_type: str
    severity: int
    notes: str
    rater_id: str
    created_at: str


class BaselineResponse(BaseModel):
    """Baseline period response model."""

    id: str
    start_date: date
    end_date: date
    notes: str
    rater_id: str
    created_at: str


class LabelStatsResponse(BaseModel):
    """Label statistics response."""

    total_episodes: int
    episodes_by_type: dict[str, int]
    total_baselines: int
    date_range: dict[str, str] | None
    raters: list[str]
    avg_severity: float | None


# Initialize repository
repository = SQLiteEpisodeRepository(db_path="labels.db")


@router.post("/episodes", response_model=EpisodeResponse)
async def create_episode(request: EpisodeCreateRequest) -> EpisodeResponse:
    """Create a new mood episode label."""
    try:
        labeler = EpisodeLabeler()

        # Add episode
        labeler.add_episode(
            start_date=request.start_date,
            end_date=request.end_date or request.start_date,
            episode_type=request.episode_type,
            severity=request.severity,
            notes=request.notes,
            rater_id=request.rater_id,
        )

        # Save to repository
        repository.save_labeler(labeler)

        # Return the created episode
        episode = labeler.episodes[-1]
        return EpisodeResponse(
            id=str(len(labeler.episodes)),
            start_date=episode["start_date"],
            end_date=episode["end_date"],
            episode_type=episode["episode_type"],
            severity=episode["severity"],
            notes=episode["notes"],
            rater_id=episode["rater_id"],
            created_at=episode.get("created_at", ""),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/baselines", response_model=BaselineResponse)
async def create_baseline(request: BaselineCreateRequest) -> BaselineResponse:
    """Create a new baseline period."""
    try:
        labeler = EpisodeLabeler()

        # Add baseline
        labeler.add_baseline(
            start_date=request.start_date,
            end_date=request.end_date,
            notes=request.notes,
            rater_id=request.rater_id,
        )

        # Save to repository
        repository.save_labeler(labeler)

        # Return created baseline
        baseline = labeler.baseline_periods[-1]
        return BaselineResponse(
            id=str(len(labeler.baseline_periods)),
            start_date=baseline["start_date"],
            end_date=baseline["end_date"],
            notes=baseline["notes"],
            rater_id=baseline["rater_id"],
            created_at=baseline.get("created_at", ""),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/episodes", response_model=list[EpisodeResponse])
async def list_episodes(
    rater_id: str | None = None,
    episode_type: str | None = None,
    limit: int = 100,
) -> list[EpisodeResponse]:
    """List mood episodes with optional filtering."""
    try:
        labeler = EpisodeLabeler()
        repository.load_into_labeler(labeler)

        episodes = labeler.episodes

        # Apply filters
        if rater_id:
            episodes = [ep for ep in episodes if ep["rater_id"] == rater_id]
        if episode_type:
            episodes = [ep for ep in episodes if ep["episode_type"] == episode_type]

        # Apply limit
        episodes = episodes[:limit]

        return [
            EpisodeResponse(
                id=str(i + 1),
                start_date=ep["start_date"],
                end_date=ep["end_date"],
                episode_type=ep["episode_type"],
                severity=ep["severity"],
                notes=ep["notes"],
                rater_id=ep["rater_id"],
                created_at=ep.get("created_at", ""),
            )
            for i, ep in enumerate(episodes)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/baselines", response_model=list[BaselineResponse])
async def list_baselines(
    rater_id: str | None = None,
    limit: int = 100,
) -> list[BaselineResponse]:
    """List baseline periods with optional filtering."""
    try:
        labeler = EpisodeLabeler()
        repository.load_into_labeler(labeler)

        baselines = labeler.baseline_periods

        # Apply filters
        if rater_id:
            baselines = [bp for bp in baselines if bp["rater_id"] == rater_id]

        # Apply limit
        baselines = baselines[:limit]

        return [
            BaselineResponse(
                id=str(i + 1),
                start_date=bp["start_date"],
                end_date=bp["end_date"],
                notes=bp["notes"],
                rater_id=bp["rater_id"],
                created_at=bp.get("created_at", ""),
            )
            for i, bp in enumerate(baselines)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats", response_model=LabelStatsResponse)
async def get_label_stats() -> LabelStatsResponse:
    """Get statistics about labeled data."""
    try:
        labeler = EpisodeLabeler()
        repository.load_into_labeler(labeler)

        # Count episodes by type
        episodes_by_type: dict[str, int] = {}
        for episode in labeler.episodes:
            ep_type = episode["episode_type"]
            episodes_by_type[ep_type] = episodes_by_type.get(ep_type, 0) + 1

        # Get unique raters
        raters = list(
            set(
                [ep["rater_id"] for ep in labeler.episodes]
                + [bp["rater_id"] for bp in labeler.baseline_periods]
            )
        )

        # Calculate average severity
        severities = [ep["severity"] for ep in labeler.episodes if "severity" in ep]
        avg_severity = sum(severities) / len(severities) if severities else None

        # Get date range
        all_dates = []
        for ep in labeler.episodes:
            all_dates.extend([ep["start_date"], ep["end_date"]])
        for bp in labeler.baseline_periods:
            all_dates.extend([bp["start_date"], bp["end_date"]])

        date_range = None
        if all_dates:
            date_range = {
                "earliest": str(min(all_dates)),
                "latest": str(max(all_dates)),
            }

        return LabelStatsResponse(
            total_episodes=len(labeler.episodes),
            episodes_by_type=episodes_by_type,
            total_baselines=len(labeler.baseline_periods),
            date_range=date_range,
            raters=raters,
            avg_severity=avg_severity,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: int) -> dict[str, str]:
    """Delete an episode by ID."""
    try:
        labeler = EpisodeLabeler()
        repository.load_into_labeler(labeler)

        if episode_id < 1 or episode_id > len(labeler.episodes):
            raise HTTPException(status_code=404, detail="Episode not found")

        # Remove episode (1-indexed)
        labeler.episodes.pop(episode_id - 1)

        # Save updated labeler
        repository.save_labeler(labeler)

        return {"message": f"Episode {episode_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/export")
async def export_labels() -> dict[str, Any]:
    """Export all labels in training-ready format."""
    try:
        labeler = EpisodeLabeler()
        repository.load_into_labeler(labeler)
        df = labeler.to_dataframe()

        # Convert to dict format
        return {
            "episodes": df.to_dict("records"),
            "total_count": len(df),
            "export_timestamp": str(date.today()),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}") from e
