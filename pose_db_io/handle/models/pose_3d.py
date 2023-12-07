from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from typing_extensions import Annotated
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer, UUID4
from pydantic.functional_validators import AfterValidator


class KeypointsFormatEnum(Enum):
    mpii_15 = "mpii-15"
    mpii_16 = "mpii-16"
    coco_17 = "coco-17"
    coco_18 = "coco-18"
    body_25 = "body-25"
    halpe_133 = "halpe-133"
    halpe_136 = "halpe-136"


class PosePairScoreDistanceMethodEnum(Enum):
    pixels = "pixels"
    image_frac = "image_frac"
    threed = "3d"


def rounded_float(v: float) -> float:
    return round(v, 3)


RoundedFloat = Annotated[float, AfterValidator(rounded_float)]


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
    ]
    min_keypoint_quality: Optional[RoundedFloat]
    min_num_keypoints: Optional[int]
    min_pose_quality: Optional[RoundedFloat]
    min_pose_pair_score: Optional[RoundedFloat]
    max_pose_pair_score: Optional[RoundedFloat]
    pose_pair_score_distance_method: PosePairScoreDistanceMethodEnum
    pose_3d_graph_initial_edge_threshold: int
    pose_3d_graph_max_dispersion: RoundedFloat

    @field_serializer("classroom_date")
    def serialize_classroom_date(self, dt: date, _info):
        return dt.strftime("%Y-%m-%d")


class Pose3dOutput(BaseModel):
    keypoints: Tuple[Tuple[RoundedFloat, RoundedFloat, RoundedFloat], ...]  # ((x, y, z), ...)


class Pose3d(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    id: UUID4 = Field(default_factory=uuid.uuid4)
    timestamp: datetime
    metadata: Pose3dMetadata
    pose: Pose3dOutput
    pose_2d_ids: Tuple[UUID4, ...]
