from typing import List
import datetime
import uuid
import collections

import numpy as np
import pandas as pd

from pymongo import InsertOne, MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import BulkWriteError

import pose_db_io.config
from pose_db_io.log import logger

from .models.pose_2d import Pose2d
from .models.pose_3d import Pose3d



class PoseHandle:
    def __init__(self, db_uri: str = None):
        if db_uri is None:
            db_uri = pose_db_io.config.Settings().MONGO_POSE_URI

        self.client: MongoClient = MongoClient(db_uri, uuidRepresentation="standard")
        self.db: MongoDatabase = self.client["poses"]
        self.poses_2d_collection: MongoCollection = self.db["poses_2d"]
        self.poses_3d_collection: MongoCollection = self.db["poses_3d"]

    def insert_poses_2d(self, pose_2d_batch: List[Pose2d]):
        bulk_requests = list(map(lambda p: InsertOne(p.model_dump()), pose_2d_batch))
        try:
            logger.debug(
                f"Inserting {len(bulk_requests)} into Mongo poses_2d database..."
            )
            self.poses_2d_collection.bulk_write(bulk_requests, ordered=False)
            logger.debug(
                f"Successfully wrote {len(bulk_requests)} records into Mongo poses_2d database..."
            )
        except BulkWriteError as e:
            logger.error(
                f"Failed writing {len(bulk_requests)} records to Mongo poses_2d database: {e}"
            )

    def generate_poses_2d_find_iterator(
        self,
        inference_run_ids=None,
        environment_id=None,
        camera_ids=None,
        start=None,
        end=None,
        database_tzinfo=None,
    ):
        if database_tzinfo is None:
            database_tzinfo = datetime.timezone.utc
        query_dict = generate_pose_2d_query_dict(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            camera_ids=camera_ids,
            start=start,
            end=end,
            database_tzinfo=database_tzinfo,
        )
        find_iterator = self.poses_2d_collection.find(query_dict)
        return find_iterator

    def fetch_poses_2d(
        self,
        inference_run_ids=None,
        environment_id=None,
        camera_ids=None,
        start=None,
        end=None,
        database_tzinfo=None,
    ):
        if database_tzinfo is None:
            database_tzinfo = datetime.timezone.utc
        find_iterator = self.generate_poses_2d_find_iterator(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            camera_ids=camera_ids,
            start=start,
            end=end,
            database_tzinfo=database_tzinfo,
        )
        poses_2d_list = list()
        for pose_2d_raw in find_iterator:
            pose_data_array = np.asarray(pose_2d_raw['pose']['keypoints'])
            keypoint_coordinates = pose_data_array[:, :2]
            keypoint_visibility = pose_data_array[:, 2]
            keypoint_quality = pose_data_array[:, 3]
            bounding_box_array = np.asarray(pose_2d_raw['bbox']['bbox'])
            bounding_box = bounding_box_array[:4]
            bounding_box_quality = bounding_box_array[4]
            poses_2d_list.append(collections.OrderedDict((
                ('pose_2d_id', str(pose_2d_raw['id'])),
                ('timestamp', pose_2d_raw['timestamp'].replace(tzinfo=database_tzinfo).astimezone(datetime.timezone.utc)),
                ('camera_id', str(pose_2d_raw['metadata']['camera_device_id'])),
                ('keypoint_coordinates_2d', keypoint_coordinates),
                ('keypoint_quality_2d', keypoint_quality),
                ('pose_quality', None),
                ('keypoint_visibility_2d', keypoint_visibility),
                ('bounding_box', bounding_box),
                ('bounding_box_quality', bounding_box_quality),
                ('bounding_box_format', pose_2d_raw['metadata']['bounding_box_format']),
                ('keypoints_format', pose_2d_raw['metadata']['keypoints_format']),
                ('inference_run_id', str(pose_2d_raw['metadata']['inference_run_id'])),
                ('inference_run_created_at', pose_2d_raw['metadata']['inference_run_created_at'].replace(tzinfo=database_tzinfo).astimezone(datetime.timezone.utc)),
            )))
        poses_2d = (
            pd.DataFrame(poses_2d_list)
            .sort_values('timestamp')
            .set_index('pose_2d_id')
        )
        return poses_2d

    def cleanup(self):
        if self.client is not None:
            self.client.close()

    def __del__(self):
        self.cleanup()

def generate_pose_2d_query_dict(
    inference_run_ids=None,
    environment_id=None,
    camera_ids=None,
    start=None,
    end=None,
    database_tzinfo=None,
):
    if database_tzinfo is None:
        database_tzinfo = datetime.timezone.utc
    query_dict = dict()
    if inference_run_ids is not None:
        query_dict['metadata.inference_run_id'] = {"$in": [uuid.UUID(inference_run_id) for inference_run_id in inference_run_ids]}
    if environment_id is not None:
        query_dict['metadata.environment_id'] = uuid.UUID(environment_id)
    if camera_ids is not None:
        query_dict['metadata.camera_device_id'] = {"$in": [uuid.UUID(camera_id) for camera_id in camera_ids]}
    if start is not None or end is not None:
        timestamp_qualifier_dict = dict()
        if start is not None:
            timestamp_qualifier_dict['$gte'] = start.astimezone(database_tzinfo).replace(tzinfo=None)
        if end is not None:
            timestamp_qualifier_dict['$lt'] = end.astimezone(database_tzinfo).replace(tzinfo=None)
        query_dict['timestamp'] = timestamp_qualifier_dict
    return query_dict
