from typing import List, Optional, Union
import datetime
import uuid
import collections

import numpy as np
import pandas as pd

from pymongo import InsertOne, MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import BulkWriteError

from pose_db_io.config import Settings
from pose_db_io.log import logger

from .models.pose_2d import Pose2d
from .models.pose_3d import Pose3d


class PoseHandle:
    def __init__(self, db_uri: str = None):
        if db_uri is None:
            db_uri = Settings().MONGO_POSE_URI

        self.client: MongoClient = MongoClient(db_uri, uuidRepresentation="standard", tz_aware=True)
        self.db: MongoDatabase = self.client.get_database("poses")
        self.poses_2d_collection: MongoCollection = self.db.get_collection("poses_2d")
        self.poses_3d_collection: MongoCollection = self.db.get_collection("poses_3d")

    def insert_poses_2d(self, pose_2d_batch: List[Pose2d]):
        bulk_requests = list(map(lambda p: InsertOne(p.model_dump()), pose_2d_batch))
        try:
            logger.debug(f"Inserting {len(bulk_requests)} into Mongo poses_2d database...")
            self.poses_2d_collection.bulk_write(bulk_requests, ordered=False)
            logger.debug(f"Successfully wrote {len(bulk_requests)} records into Mongo poses_2d database...")
        except BulkWriteError as e:
            logger.error(f"Failed writing {len(bulk_requests)} records to Mongo poses_2d database: {e}")

    def fetch_poses_2d_dataframe(
        self,
        inference_run_ids=None,
        environment_id=None,
        camera_ids=None,
        start=None,
        end=None,
    ):
        find_iterator = self.generate_poses_2d_find_iterator(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            camera_ids=camera_ids,
            start=start,
            end=end,
        )
        poses_2d_list = []
        for pose_2d_raw in find_iterator:
            pose_data_array = np.asarray(pose_2d_raw["pose"]["keypoints"])
            keypoint_coordinates = pose_data_array[:, :2]
            keypoint_visibility = pose_data_array[:, 2]
            keypoint_quality = pose_data_array[:, 3]
            bounding_box_array = np.asarray(pose_2d_raw["bbox"]["bbox"])
            bounding_box = bounding_box_array[:4]
            bounding_box_quality = bounding_box_array[4]
            pose_quality = np.nanmean(keypoint_quality)
            poses_2d_list.append(
                collections.OrderedDict(
                    (
                        ("pose_2d_id", str(pose_2d_raw["id"])),
                        ("timestamp", pose_2d_raw["timestamp"]),
                        ("camera_id", str(pose_2d_raw["metadata"]["camera_device_id"])),
                        ("keypoint_coordinates_2d", keypoint_coordinates),
                        ("keypoint_quality_2d", keypoint_quality),
                        ("pose_quality_2d", pose_quality),
                        ("keypoint_visibility_2d", keypoint_visibility),
                        ("bounding_box", bounding_box),
                        ("bounding_box_quality", bounding_box_quality),
                        ("bounding_box_format", pose_2d_raw["metadata"]["bounding_box_format"]),
                        ("keypoints_format", pose_2d_raw["metadata"]["keypoints_format"]),
                        ("inference_run_id", str(pose_2d_raw["metadata"]["inference_run_id"])),
                        ("inference_run_created_at", pose_2d_raw["metadata"]["inference_run_created_at"]),
                    )
                )
            )

        poses_2d = None
        if len(poses_2d_list) > 0:
            poses_2d = pd.DataFrame(poses_2d_list).sort_values(["timestamp", "camera_id"]).set_index("pose_2d_id")
        return poses_2d

    def fetch_poses_2d_objects(
        self,
        inference_run_ids=None,
        environment_id=None,
        camera_ids=None,
        start=None,
        end=None,
    ):
        find_iterator = self.generate_poses_2d_find_iterator(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            camera_ids=camera_ids,
            start=start,
            end=end,
        )

        poses_2d_list = []
        for pose_2d_raw in find_iterator:
            poses_2d_list.append(Pose2d(**pose_2d_raw))
        return poses_2d_list

    def generate_poses_2d_find_iterator(
        self, inference_run_ids=None, environment_id=None, camera_ids=None, start=None, end=None
    ):
        query_dict = self.generate_pose_2d_query_dict(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            camera_ids=camera_ids,
            start=start,
            end=end,
        )
        find_iterator = self.poses_2d_collection.find(query_dict)
        return find_iterator

    def fetch_pose_2d_coverage_dataframe_by_environment_id(self, environment_id: Union[str, uuid.UUID]):
        pose_2d_coverage_cursor = self.poses_2d_collection.aggregate(
            [
                {
                    "$match": {
                        "metadata.environment_id": environment_id
                        if isinstance(environment_id, uuid.UUID)
                        else uuid.UUID(environment_id)
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "metadata": {
                                "inference_run_created_at": "$metadata.inference_run_created_at",
                                "inference_run_id": "$metadata.inference_run_id",
                            }
                        },
                        "inference_run_id": {"$first": "$metadata.inference_run_id"},
                        "inference_run_created_at": {"$first": "$metadata.inference_run_created_at"},
                        "environment_id": {"$first": "$metadata.environment_id"},
                        "count": {"$sum": 1},
                        "start": {"$min": "$timestamp"},
                        "end": {"$max": "$timestamp"},
                    }
                },
                {"$sort": {"start": -1, "inference_run_created_at": -1}},
            ]
        )
        pose_2d_coverage_list = []
        for item in pose_2d_coverage_cursor:
            pose_2d_coverage_list.append(
                collections.OrderedDict(
                    (
                        ("inference_run_id", str(item["inference_run_id"])),
                        ("inference_run_created_at", item["inference_run_created_at"]),
                        ("environment_id", str(item["environment_id"])),
                        ("count", item["count"]),
                        ("start", item["start"]),
                        ("end", item["end"]),
                    )
                )
            )
        df_pose_2d_coverage = None
        if len(pose_2d_coverage_list) > 0:
            df_pose_2d_coverage = pd.DataFrame(pose_2d_coverage_list)

        return df_pose_2d_coverage

    @staticmethod
    def generate_pose_2d_query_dict(
        inference_run_ids: Optional[Union[List[str], List[uuid.UUID]]] = None,
        environment_id: Optional[Union[str, uuid.UUID]] = None,
        camera_ids: Optional[Union[List[str], List[uuid.UUID]]] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
    ):
        database_tzinfo = datetime.timezone.utc

        if start is not None and start.tzinfo is None:
            raise ValueError(
                "generate_pose_2d_query_dict 'start' attribute must be None or timezone aware datetime object"
            )

        if end is not None and end.tzinfo is None:
            raise ValueError(
                "generate_pose_2d_query_dict 'end' attribute must be None or timezone aware datetime object"
            )

        query_dict = {}
        if inference_run_ids is not None:
            query_dict["metadata.inference_run_id"] = {
                "$in": [uuid.UUID(inference_run_id) for inference_run_id in inference_run_ids]
            }
        if environment_id is not None:
            query_dict["metadata.environment_id"] = uuid.UUID(environment_id)
        if camera_ids is not None:
            query_dict["metadata.camera_device_id"] = {"$in": [uuid.UUID(camera_id) for camera_id in camera_ids]}
        if start is not None or end is not None:
            timestamp_qualifier_dict = {}
            if start is not None:
                timestamp_qualifier_dict["$gte"] = start.astimezone(database_tzinfo)
            if end is not None:
                timestamp_qualifier_dict["$lt"] = end.astimezone(database_tzinfo)
            query_dict["timestamp"] = timestamp_qualifier_dict
        return query_dict

    def insert_poses_3d(self, pose_3d_batch: List[Pose3d]):
        bulk_requests = list(map(lambda p: InsertOne(p.model_dump()), pose_3d_batch))
        try:
            logger.debug(f"Inserting {len(bulk_requests)} into Mongo poses_3d database...")
            self.poses_3d_collection.bulk_write(bulk_requests, ordered=False)
            logger.debug(f"Successfully wrote {len(bulk_requests)} records into Mongo poses_3d database...")
        except BulkWriteError as e:
            logger.error(f"Failed writing {len(bulk_requests)} records to Mongo poses_3d database: {e}")

    def fetch_poses_3d_dataframe(
        self,
        inference_run_ids=None,
        environment_id=None,
        start=None,
        end=None,
    ):
        find_iterator = self.generate_poses_3d_find_iterator(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            start=start,
            end=end,
        )
        poses_3d_list = []
        for pose_3d_raw in find_iterator:
            keypoint_coordinates = np.asarray(pose_3d_raw["pose"]["keypoints"])
            poses_3d_list.append(
                collections.OrderedDict(
                    (
                        ("pose_3d_id", str(pose_3d_raw["id"])),
                        ("timestamp", pose_3d_raw["timestamp"]),
                        ("keypoint_coordinates_3d", keypoint_coordinates),
                        ("pose_2d_ids", [str(pose_2d_id) for pose_2d_id in pose_3d_raw["pose_2d_ids"]]),
                        ("keypoints_format", pose_3d_raw["metadata"]["keypoints_format"]),
                        ("inference_run_id", str(pose_3d_raw["metadata"]["inference_run_id"])),
                        ("inference_run_created_at", pose_3d_raw["metadata"]["inference_run_created_at"]),
                    )
                )
            )

        poses_3d = None
        if len(poses_3d_list) > 0:
            poses_3d = pd.DataFrame(poses_3d_list).sort_values("timestamp").set_index("pose_3d_id")
        return poses_3d

    def fetch_poses_3d_objects(
        self,
        inference_run_ids=None,
        environment_id=None,
        start=None,
        end=None,
    ):
        find_iterator = self.generate_poses_3d_find_iterator(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            start=start,
            end=end,
        )

        poses_3d_list = []
        for pose_3d_raw in find_iterator:
            poses_3d_list.append(Pose3d(**pose_3d_raw))
        return poses_3d_list

    def generate_poses_3d_find_iterator(self, inference_run_ids=None, environment_id=None, start=None, end=None):
        query_dict = self.generate_pose_3d_query_dict(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            start=start,
            end=end,
        )
        find_iterator = self.poses_3d_collection.find(query_dict)
        return find_iterator

    @staticmethod
    def generate_pose_3d_query_dict(
        inference_run_ids: Optional[Union[List[str], List[uuid.UUID]]] = None,
        environment_id: Optional[Union[str, uuid.UUID]] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
    ):
        database_tzinfo = datetime.timezone.utc

        if start is not None and start.tzinfo is None:
            raise ValueError(
                "generate_pose_2d_query_dict 'start' attribute must be None or timezone aware datetime object"
            )

        if end is not None and end.tzinfo is None:
            raise ValueError(
                "generate_pose_2d_query_dict 'end' attribute must be None or timezone aware datetime object"
            )

        query_dict = {}
        if inference_run_ids is not None:
            query_dict["metadata.inference_run_id"] = {
                "$in": [uuid.UUID(inference_run_id) for inference_run_id in inference_run_ids]
            }
        if environment_id is not None:
            query_dict["metadata.environment_id"] = uuid.UUID(environment_id)
        if start is not None or end is not None:
            timestamp_qualifier_dict = {}
            if start is not None:
                timestamp_qualifier_dict["$gte"] = start.astimezone(database_tzinfo)
            if end is not None:
                timestamp_qualifier_dict["$lt"] = end.astimezone(database_tzinfo)
            query_dict["timestamp"] = timestamp_qualifier_dict
        return query_dict

    def cleanup(self):
        if self.client is not None:
            self.client.close()

    def __del__(self):
        self.cleanup()
