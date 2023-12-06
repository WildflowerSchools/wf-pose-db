from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from typing_extensions import Annotated
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer, UUID4
from pydantic.functional_validators import AfterValidator


class PosePairScoreDistanceMethodEnum(Enum):
    pixels = 'pixels'
    image_frac = 'image_frac'
    threed = '3d'

class Pose3dMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    inference_run_id: UUID4
    inference_run_created_at: datetime
    environment_id: UUID4
    classroom_date: date
    coordinate_space_id: UUID4
    pose_model_id: UUID4
    keypoints_format: KeypointsFormatEnum
    pose_3d_limits: Tuple[
        Tuple[Tuple[RoundedFloat, RoundedFloat, RoundedFloat], ...],
        Tuple[Tuple[RoundedFloat, RoundedFloat, RoundedFloat], ...],
    ] = None
    room_x_limits: Tuple[RoundedFloat, RoundedFloat]
    room_y_limits: Tuple[RoundedFloat, RoundedFloat]
    floor_z: RoundedFloat
    foot_z_limits: Tuple[RoundedFloat, RoundedFloat]
    knee_z_limits: Tuple[RoundedFloat, RoundedFloat]
    hip_z_limits: Tuple[RoundedFloat, RoundedFloat]
    thorax_z_limits: Tuple[RoundedFloat, RoundedFloat]
    shoulder_z_limits: Tuple[RoundedFloat, RoundedFloat]
    elbow_z_limits: Tuple[RoundedFloat, RoundedFloat]
    hand_z_limits: Tuple[RoundedFloat, RoundedFloat]
    neck_z_limits: Tuple[RoundedFloat, RoundedFloat]
    head_z_limits: Tuple[RoundedFloat, RoundedFloat]
    tolerance: RoundedFloat
    min_keypoint_quality: RoundedFloat
    min_num_keypoints: int
    min_pose_quality: RoundedFloat
    min_pose_pair_score: RoundedFloat
    max_pose_pair_score: RoundedFloat
    pose_pair_score_distance_method: PosePairScoreDistanceMethodEnum
    pose_3d_graph_initial_edge_threshold: int
    pose_3d_graph_max_dispersion: RoundedFloat
    @field_serializer("classroom_date")
    def serialize_classroom_date(self, dt: date, _info):
        return dt.strftime("%Y-%m-%d")


def rounded_float(v: float) -> float:
    return round(v, 3)


RoundedFloat = Annotated[float, AfterValidator(rounded_float)]


class Pose3dOutput(BaseModel):
    keypoints: Tuple[
        Tuple[RoundedFloat, RoundedFloat, RoundedFloat], ...
    ]  # ((x, y, z), ...)


class Pose3d(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    timestamp: datetime
    metadata: Pose3dMetadata
    pose: Pose3dOutput
    pose_2d_ids: Tuple[UUID4, ...]
