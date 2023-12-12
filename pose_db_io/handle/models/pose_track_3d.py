from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from typing_extensions import Annotated
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer, UUID4
from pydantic.functional_validators import AfterValidator


def rounded_float(v: float) -> float:
    return round(v, 3)


RoundedFloat = Annotated[float, AfterValidator(rounded_float)]


class PoseTrack3dMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    inference_run_id: UUID4
    inference_run_created_at: datetime
    environment_id: UUID4
    classroom_date: date
    start: datetime
    end: datetime
    max_match_distance: Optional[RoundedFloat]
    max_iterations_since_last_match: Optional[int]
    centroid_position_initial_sd: RoundedFloat
    centroid_velocity_initial_sd: RoundedFloat
    reference_delta_t_seconds: RoundedFloat
    reference_velocity_drift: RoundedFloat
    position_observation_sd: RoundedFloat
    num_poses_per_track_min: Optional[int]

    @field_serializer("classroom_date")
    def serialize_classroom_date(self, dt: date, _info):
        return dt.strftime("%Y-%m-%d")


class PoseTrack3d(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    id: UUID4 = Field(default_factory=uuid.uuid4)
    metadata: PoseTrack3dMetadata
