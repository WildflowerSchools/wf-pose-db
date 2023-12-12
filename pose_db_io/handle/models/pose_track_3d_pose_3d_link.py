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


class PoseTrack3dPose3dLinkMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    inference_run_id: UUID4
    inference_run_created_at: datetime

class PoseTrack3dPose3dLink(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    metadata: PoseTrack3dPose3dLinkMetadata
    pose_track_3d_id: UUID4
    pose_3d_id: UUID4