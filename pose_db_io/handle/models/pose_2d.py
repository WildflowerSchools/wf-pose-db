from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from typing_extensions import Annotated
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer, UUID4
from pydantic.functional_validators import AfterValidator


class BoundingBoxFormatEnum(Enum):
    xyxy = "xyxy"
    xywh = "xywh"


class KeypointsFormatEnum(Enum):
    mpii_15 = "mpii-15"
    mpii_16 = "mpii-16"
    coco_17 = "coco-17"
    coco_18 = "coco-18"
    body_25 = "body-25"
    halpe_133 = "halpe-133"
    halpe_136 = "halpe-136"


class PoseModelConfigEnum(Enum):
    rtmpose_s_8xb256_420e_body8_256x192 = "rtmpose-s_8xb256-420e_body8-256x192"
    rtmpose_m_8xb256_420e_body8_256x192 = "rtmpose-m_8xb256-420e_body8-256x192"
    rtmpose_m_8xb256_420e_body8_384x288 = "rtmpose-m_8xb256-420e_body8-384x288"
    rtmpose_l_8xb256_420e_body8_256x192 = "rtmpose-l_8xb256-420e_body8-256x192"
    rtmpose_l_8xb256_420e_body8_384x288 = "rtmpose-l_8xb256-420e_body8-384x288"
    rtmo_l_16xb16_600e_body7_640x640 = "rtmo_l_16xb16_600e_body7_640x640"
    rtmo_m_16xb16_600e_body7_640x640 = "rtmo-m_16xb16-600e_body7-640x640"


class PoseModelCheckpointEnum(Enum):
    rtmpose_s_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_m_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_m_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_l_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504_tensorrt_dynamic_384x288_batch = (
        "rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504_tensorrt_dynamic_384x288_batch"
    )
    rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211 = (
        "rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211"
    )
    rtmo_m_16xb16_600e_body7_640x640_39e78cc4_20231211 = (
        "rtmo_m_16xb16_600e_body7_640x640_39e78cc4_20231211"
    )


class PoseModelDeploymentConfigEnum(Enum):
    tensorrt_simcc_dynamic_384x288_batch = "tensorrt_simcc_dynamic_384x288_batch"


class DetectorModelConfigEnum(Enum):
    rtmdet_nano_640_8xb32_coco_person = "rtmdet_nano_640-8xb32_coco-person"
    rtmdet_m_640_8xb32_coco_person = "rtmdet_m_640-8xb32_coco-person"


class DetectorModelCheckpointEnum(Enum):
    rtmdet_nano_8xb32_100e_coco_obj365_person_05d8511e = "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e"
    rtmdet_m_8xb32_100e_coco_obj365_person_235e8209 = "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209"
    rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640 = "rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640"
    rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_batch = "rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_batch"
    rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_fp16_batch = "rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_fp16_batch"


class DetectorModelDeploymentConfigEnum(Enum):
    tensorrt_static_640x640 = "tensorrt_static_640x640"
    tensorrt_dynamic_640x640_batch = "tensorrt_dynamic_640x640_batch"
    tensorrt_dynamic_640x640_fp16_batch = "tensorrt_dynamic_640x640_fp16_batch"


# def coerce_to_uuid(uuid_like_object):
#     return uuid.UUID(uuid_like_object)

# FlexibleUUID = Annotated[Union[UUID4, str], AfterValidator(double), AfterValidator(check_squares)]

class PoseEstimatorType(Enum):
    top_down = "top_down"
    one_stage = "one_stage"


class Pose2dMetadataCommon(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    inference_run_id: UUID4
    inference_run_created_at: datetime
    environment_id: UUID4
    classroom_date: date
    keypoints_format: KeypointsFormatEnum
    bounding_box_format: BoundingBoxFormatEnum
    pose_model_config: PoseModelConfigEnum
    pose_model_checkpoint: PoseModelCheckpointEnum
    pose_model_deployment_config: Optional[PoseModelDeploymentConfigEnum] = Field(default=None)
    detection_model_config: Optional[DetectorModelConfigEnum]
    detection_model_checkpoint: Optional[DetectorModelCheckpointEnum]
    detection_model_deployment_config: Optional[DetectorModelDeploymentConfigEnum] = Field(default=None)

    @field_serializer("classroom_date")
    def serialize_classroom_date(self, dt: date, _info):
        return dt.strftime("%Y-%m-%d")


class Pose2dMetadata(Pose2dMetadataCommon):
    camera_device_id: UUID4


def rounded_float(v: float) -> float:
    return round(v, 3)


RoundedFloat = Annotated[float, AfterValidator(rounded_float)]


class PoseOutput(BaseModel):
    keypoints: Tuple[
        Tuple[RoundedFloat, RoundedFloat, RoundedFloat, RoundedFloat], ...
    ]  # ((x, y, visibility, score), ...)


class BoundingBoxOutput(BaseModel):
    bbox: Tuple[RoundedFloat, RoundedFloat, RoundedFloat, RoundedFloat, RoundedFloat]  # (x, y, x|w, y|h, score)


class Pose2d(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    id: UUID4 = Field(default_factory=uuid.uuid4)
    timestamp: datetime
    metadata: Pose2dMetadata
    pose: PoseOutput
    bbox: BoundingBoxOutput
