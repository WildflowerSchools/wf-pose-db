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
from .models.pose_track_3d import PoseTrack3d
from .models.pose_track_3d_pose_3d_link import PoseTrack3dPose3dLink

def coerce_to_uuid(uuid_like_object):
    if isinstance(uuid_like_object, uuid.UUID):
        return(uuid_like_object)
    else:
        return(uuid.UUID(uuid_like_object))

class PoseHandle:
    def __init__(self, db_uri: str = None):
        if db_uri is None:
            db_uri = Settings().MONGO_POSE_URI

        self.client: MongoClient = MongoClient(db_uri, uuidRepresentation="standard", tz_aware=True)
        self.db: MongoDatabase = self.client.get_database("poses")
        self.poses_2d_collection: MongoCollection = self.db.get_collection("poses_2d")
        self.poses_3d_collection: MongoCollection = self.db.get_collection("poses_3d")
        self.pose_tracks_3d_collection: MongoCollection = self.db.get_collection("pose_tracks_3d")
        self.pose_track_3d_pose_3d_links_collection: MongoCollection = self.db.get_collection("pose_track_3d_pose_3d_links") 

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
        remove_inference_run_overlaps=True
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
            if remove_inference_run_overlaps:
                poses_2d = self.remove_inference_run_overlaps_dataframe(poses_2d)
        return poses_2d

    @staticmethod
    def remove_inference_run_overlaps_dataframe(poses):
        poses_without_overlaps = (
            poses
            .groupby('timestamp', group_keys=False)
            .apply(lambda x: x.loc[x['inference_run_created_at'] == x['inference_run_created_at'].max()])
            .sort_values([
                'timestamp',
                'camera_id'
            ])
        )
        return poses_without_overlaps

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
        df_pose_2d_coverage = self.fetch_coverage_dataframe_by_environment_id(
            environment_id=environment_id,
            collection=self.poses_2d_collection
        )
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
                "$in": [coerce_to_uuid(inference_run_id) for inference_run_id in inference_run_ids]
            }
        if environment_id is not None:
            query_dict["metadata.environment_id"] = coerce_to_uuid(environment_id)
        if camera_ids is not None:
            query_dict["metadata.camera_device_id"] = {"$in": [coerce_to_uuid(camera_id) for camera_id in camera_ids]}
        if start is not None or end is not None:
            timestamp_qualifier_dict = {}
            if start is not None:
                timestamp_qualifier_dict["$gte"] = start.astimezone(database_tzinfo)
            if end is not None:
                timestamp_qualifier_dict["$lt"] = end.astimezone(database_tzinfo)
            query_dict["timestamp"] = timestamp_qualifier_dict
        return query_dict

    def insert_poses_3d_dataframe(
        self,
        poses_3d,
        inference_id,
        inference_run_created_at,
        environment_id,
        classroom_date,
        coordinate_space_id,
        pose_model_id,
        keypoints_format,
        pose_3d_limits,
        min_keypoint_quality,
        min_num_keypoints,
        min_pose_quality,
        min_pose_pair_score,
        max_pose_pair_score,
        pose_pair_score_distance_method,
        pose_3d_graph_initial_edge_threshold,
        pose_3d_graph_max_dispersion,
    ):
        pose_3d_objects = self.convert_poses_3d_dataframe_to_pose3d_objects(
            poses_3d=poses_3d,
            inference_id=inference_id,
            inference_run_created_at=inference_run_created_at,
            environment_id=environment_id,
            classroom_date=classroom_date,
            coordinate_space_id=coordinate_space_id,
            pose_model_id=pose_model_id,
            keypoints_format=keypoints_format,
            pose_3d_limits=pose_3d_limits,
            min_keypoint_quality=min_keypoint_quality,
            min_num_keypoints=min_num_keypoints,
            min_pose_quality=min_pose_quality,
            min_pose_pair_score=min_pose_pair_score,
            max_pose_pair_score=max_pose_pair_score,
            pose_pair_score_distance_method=pose_pair_score_distance_method,
            pose_3d_graph_initial_edge_threshold=pose_3d_graph_initial_edge_threshold,
            pose_3d_graph_max_dispersion=pose_3d_graph_max_dispersion,
        )
        if len(pose_3d_objects) > 0:
            self.insert_poses_3d(pose_3d_batch=pose_3d_objects)

    @staticmethod
    def convert_poses_3d_dataframe_to_pose3d_objects(
        poses_3d,
        inference_id,
        inference_run_created_at,
        environment_id,
        classroom_date,
        coordinate_space_id,
        pose_model_id,
        keypoints_format,
        pose_3d_limits,
        min_keypoint_quality,
        min_num_keypoints,
        min_pose_quality,
        min_pose_pair_score,
        max_pose_pair_score,
        pose_pair_score_distance_method,
        pose_3d_graph_initial_edge_threshold,
        pose_3d_graph_max_dispersion,
    ):
        pose_3d_objects = list()
        for pose_3d_id, pose_3d_data in poses_3d.iterrows():
            pose_3d_objects.append(Pose3d(**{
                'id': pose_3d_id,
                'timestamp': pose_3d_data['timestamp'],
                'metadata': {
                    'inference_run_id': inference_id,
                    'inference_run_created_at': inference_run_created_at,
                    'environment_id': environment_id,
                    'classroom_date': classroom_date,
                    'coordinate_space_id': coordinate_space_id,
                    'pose_model_id': pose_model_id,
                    'keypoints_format': keypoints_format,
                    'pose_3d_limits': pose_3d_limits,
                    'min_keypoint_quality': min_keypoint_quality,
                    'min_num_keypoints': min_num_keypoints,
                    'min_pose_quality': min_pose_quality,
                    'min_pose_pair_score': min_pose_pair_score,
                    'max_pose_pair_score': max_pose_pair_score,
                    'pose_pair_score_distance_method': pose_pair_score_distance_method,
                    'pose_3d_graph_initial_edge_threshold': pose_3d_graph_initial_edge_threshold,
                    'pose_3d_graph_max_dispersion': pose_3d_graph_max_dispersion,
                },
                'pose': {'keypoints': pose_3d_data['keypoint_coordinates_3d']},
                'pose_2d_ids': pose_3d_data['pose_2d_ids']            
            }))
        return pose_3d_objects

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

    def fetch_pose_3d_coverage_dataframe_by_environment_id(self, environment_id: Union[str, uuid.UUID]):
        df_pose_3d_coverage = self.fetch_coverage_dataframe_by_environment_id(
            environment_id=environment_id,
            collection=self.poses_3d_collection
        )
        return df_pose_3d_coverage

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
                "$in": [coerce_to_uuid(inference_run_id) for inference_run_id in inference_run_ids]
            }
        if environment_id is not None:
            query_dict["metadata.environment_id"] = coerce_to_uuid(environment_id)
        if start is not None or end is not None:
            timestamp_qualifier_dict = {}
            if start is not None:
                timestamp_qualifier_dict["$gte"] = start.astimezone(database_tzinfo)
            if end is not None:
                timestamp_qualifier_dict["$lt"] = end.astimezone(database_tzinfo)
            query_dict["timestamp"] = timestamp_qualifier_dict
        return query_dict

    def create_pose_tracks_3d(
        self,
        pose_tracks_output,
        inference_id,
        inference_run_created_at,
        environment_id,
        classroom_date,
        max_match_distance,
        max_iterations_since_last_match,
        centroid_position_initial_sd,
        centroid_velocity_initial_sd,
        reference_delta_t_seconds,
        reference_velocity_drift,
        position_observation_sd,
        num_poses_per_track_min,
    ):
        pose_track_3d_objects = self.convert_pose_tracks_output_to_pose_track_3d_objects(
            pose_tracks_output=pose_tracks_output,
            inference_id=inference_id,
            inference_run_created_at=inference_run_created_at,
            environment_id=environment_id,
            classroom_date=classroom_date,
            max_match_distance=max_match_distance,
            max_iterations_since_last_match=max_iterations_since_last_match,
            centroid_position_initial_sd=centroid_position_initial_sd,
            centroid_velocity_initial_sd=centroid_velocity_initial_sd,
            reference_delta_t_seconds=reference_delta_t_seconds,
            reference_velocity_drift=reference_velocity_drift,
            position_observation_sd=position_observation_sd,
            num_poses_per_track_min=num_poses_per_track_min,
        )
        pose_track_3d_pose_3d_link_objects = self.convert_pose_tracks_output_to_pose_track_3d_pose_3d_link_objects(
            pose_tracks_output=pose_tracks_output,
            inference_id=inference_id,
            inference_run_created_at=inference_run_created_at,
        )
        if len(pose_track_3d_objects) > 0:
            self.insert_pose_tracks_3d(pose_track_3d_batch=pose_track_3d_objects)
        if len(pose_track_3d_pose_3d_link_objects) > 0:
            self.insert_pose_track_3d_pose_3d_links(pose_track_3d_pose_3d_link_batch=pose_track_3d_pose_3d_link_objects)

    @staticmethod
    def convert_pose_tracks_output_to_pose_track_3d_objects(
        pose_tracks_output,
        inference_id,
        inference_run_created_at,
        environment_id,
        classroom_date,
        max_match_distance,
        max_iterations_since_last_match,
        centroid_position_initial_sd,
        centroid_velocity_initial_sd,
        reference_delta_t_seconds,
        reference_velocity_drift,
        position_observation_sd,
        num_poses_per_track_min,
    ):
        pose_track_3d_objects = list()
        for pose_track_3d_id, pose_track_info in pose_tracks_output.items():
            pose_track_3d_objects.append(PoseTrack3d(**{
                'id': pose_track_3d_id,
                'metadata':{
                    'inference_run_id': inference_id,
                    'inference_run_created_at': inference_run_created_at,
                    'environment_id': environment_id,
                    'classroom_date': classroom_date,
                    'start': pose_track_info['start'],
                    'end': pose_track_info['end'],
                    'max_match_distance': max_match_distance,
                    'max_iterations_since_last_match': max_iterations_since_last_match,
                    'centroid_position_initial_sd': centroid_position_initial_sd,
                    'centroid_velocity_initial_sd': centroid_velocity_initial_sd,
                    'reference_delta_t_seconds': reference_delta_t_seconds,
                    'reference_velocity_drift': reference_velocity_drift,
                    'position_observation_sd': position_observation_sd,
                    'num_poses_per_track_min': num_poses_per_track_min,
                }
            }))
        return pose_track_3d_objects

    @staticmethod
    def convert_pose_tracks_output_to_pose_track_3d_pose_3d_link_objects(
        pose_tracks_output,
        inference_id,
        inference_run_created_at,
    ):
        pose_track_3d_pose_3d_link_objects = list()
        for pose_track_3d_id, pose_track_info in pose_tracks_output.items():
            for pose_3d_id in pose_track_info['pose_3d_ids']:
                pose_track_3d_pose_3d_link_objects.append(PoseTrack3dPose3dLink(**{
                    'metadata': {
                        'inference_run_id': inference_id,
                        'inference_run_created_at': inference_run_created_at,
                    },
                    'pose_track_3d_id': pose_track_3d_id,
                    'pose_3d_id': pose_3d_id
                }))
        return pose_track_3d_pose_3d_link_objects

    def insert_pose_tracks_3d(self, pose_track_3d_batch: List[PoseTrack3d]):
        bulk_requests = list(map(lambda p: InsertOne(p.model_dump()), pose_track_3d_batch))
        try:
            logger.debug(f"Inserting {len(bulk_requests)} into Mongo pose_tracks_3d database...")
            self.pose_tracks_3d_collection.bulk_write(bulk_requests, ordered=False)
            logger.debug(f"Successfully wrote {len(bulk_requests)} records into Mongo pose_tracks_3d database...")
        except BulkWriteError as e:
            logger.error(f"Failed writing {len(bulk_requests)} records to Mongo pose_tracks_3d database: {e}")

    def insert_pose_track_3d_pose_3d_links(self, pose_track_3d_pose_3d_link_batch: List[PoseTrack3dPose3dLink]):
        bulk_requests = list(map(lambda p: InsertOne(p.model_dump()), pose_track_3d_pose_3d_link_batch))
        try:
            logger.debug(f"Inserting {len(bulk_requests)} into Mongo pose_track_3d_pose_3d_links database...")
            self.pose_track_3d_pose_3d_links_collection.bulk_write(bulk_requests, ordered=False)
            logger.debug(f"Successfully wrote {len(bulk_requests)} records into Mongo pose_track_3d_pose_3d_links database...")
        except BulkWriteError as e:
            logger.error(f"Failed writing {len(bulk_requests)} records to Mongo pose_track_3d_pose_3d_links database: {e}")

    def fetch_pose_tracks_3d_dataframe(
        self,
        inference_run_ids=None,
        environment_id=None,
        start=None,
        end=None,
    ):
        find_iterator = self.generate_pose_tracks_3d_find_iterator(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            start=start,
            end=end,
        )
        pose_tracks_3d_list = []
        for pose_track_3d_raw in find_iterator:
            pose_tracks_3d_list.append(
                collections.OrderedDict(
                    (
                        ("pose_track_3d_id", str(pose_track_3d_raw["id"])),
                        ("start", pose_track_3d_raw["metadata"]["start"]),
                        ("end", pose_track_3d_raw["metadata"]["end"]),
                        ("inference_run_id", str(pose_track_3d_raw["metadata"]["inference_run_id"])),
                        ("inference_run_created_at", pose_track_3d_raw["metadata"]["inference_run_created_at"]),
                    )
                )
            )

        pose_tracks_3d = None
        if len(pose_tracks_3d_list) > 0:
            pose_tracks_3d = pd.DataFrame(pose_tracks_3d_list).sort_values("start").set_index("pose_track_3d_id")
        return pose_tracks_3d

    def fetch_pose_tracks_3d_objects(
        self,
        inference_run_ids=None,
        environment_id=None,
        start=None,
        end=None,
    ):
        find_iterator = self.generate_pose_tracks_3d_find_iterator(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            start=start,
            end=end,
        )
        pose_tracks_3d_list = []
        for pose_track_3d_raw in find_iterator:
            pose_tracks_3d_list.append(PoseTrack3d(**pose_track_3d_raw))
        return pose_tracks_3d_list

    def generate_pose_tracks_3d_find_iterator(self, inference_run_ids=None, environment_id=None, start=None, end=None):
        query_dict = self.generate_pose_track_3d_query_dict(
            inference_run_ids=inference_run_ids,
            environment_id=environment_id,
            start=start,
            end=end,
        )
        find_iterator = self.pose_tracks_3d_collection.find(query_dict)
        return find_iterator

    @staticmethod
    def generate_pose_track_3d_query_dict(
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
                "$in": [coerce_to_uuid(inference_run_id) for inference_run_id in inference_run_ids]
            }
        if environment_id is not None:
            query_dict["metadata.environment_id"] = coerce_to_uuid(environment_id)
        if start is not None:
            query_dict["metadata.end"] = {
                "gte": start
            }
        if end is not None:
            query_dict["metadata.start"] = {
                "lt": end
            }
        return query_dict

    def fetch_pose_track_3d_pose_3d_links_dataframe(
        self,
        inference_run_ids=None,
        pose_track_3d_ids=None,
    ):
        find_iterator = self.generate_pose_track_3d_pose_3d_links_find_iterator(
            inference_run_ids=inference_run_ids,
            pose_track_3d_ids=pose_track_3d_ids,
        )
        pose_track_3d_pose_3d_links_list = []
        for pose_track_3d_pose_3d_link_raw in find_iterator:
            pose_track_3d_pose_3d_links_list.append(
                collections.OrderedDict(
                    (
                        ("pose_track_3d_id", str(pose_track_3d_pose_3d_link_raw["pose_track_3d_id"])),
                        ("pose_3d_id", str(pose_track_3d_pose_3d_link_raw["pose_3d_id"])),
                        ("inference_run_id", str(pose_track_3d_pose_3d_link_raw["metadata"]["inference_run_id"])),
                        ("inference_run_created_at", pose_track_3d_pose_3d_link_raw["metadata"]["inference_run_created_at"]),
                    )
                )
            )

        pose_track_3d_pose_3d_links = None
        if len(pose_track_3d_pose_3d_links_list) > 0:
            pose_track_3d_pose_3d_links = pd.DataFrame(pose_track_3d_pose_3d_links_list).set_index("pose_3d_id")
        return pose_track_3d_pose_3d_links

    def fetch_pose_track_3d_pose_3d_links_objects(
        self,
        inference_run_ids=None,
        pose_track_3d_ids=None,
    ):
        find_iterator = self.generate_pose_track_3d_pose_3d_links_find_iterator(
            inference_run_ids=inference_run_ids,
            pose_track_3d_ids=pose_track_3d_ids,
        )
        pose_track_3d_pose_3d_links_list = []
        for pose_track_3d_pose_3d_link_raw in find_iterator:
            pose_track_3d_pose_3d_links_list.append(PoseTrack3dPose3dLink(**pose_track_3d_pose_3d_link_raw))
        return pose_track_3d_pose_3d_links_list

    def generate_pose_track_3d_pose_3d_links_find_iterator(self, inference_run_ids=None, pose_track_3d_ids=None):
        query_dict = self.generate_pose_track_3d_pose_3d_link_query_dict(
            inference_run_ids=inference_run_ids,
            pose_track_3d_ids=pose_track_3d_ids,
        )
        find_iterator = self.pose_track_3d_pose_3d_links_collection.find(query_dict)
        return find_iterator

    @staticmethod
    def generate_pose_track_3d_pose_3d_link_query_dict(
        inference_run_ids: Optional[Union[List[str], List[uuid.UUID]]] = None,
        pose_track_3d_ids:  Optional[Union[List[str], List[uuid.UUID]]] = None,
    ):
        query_dict = {}
        if inference_run_ids is not None:
            query_dict["metadata.inference_run_id"] = {
                "$in": [coerce_to_uuid(inference_run_id) for inference_run_id in inference_run_ids]
            }
        if pose_track_3d_ids is not None:
            query_dict["pose_track_3d_id"] = {
                "$in": [coerce_to_uuid(pose_track_3d_id) for pose_track_3d_id in pose_track_3d_ids]
            }
        return query_dict

    @staticmethod
    def fetch_coverage_dataframe_by_environment_id(environment_id: Union[str, uuid.UUID], collection):
        pose_coverage_cursor = collection.aggregate(
            [
                {
                    "$match": {
                        "metadata.environment_id": coerce_to_uuid(environment_id)
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
                        "classroom_date": {"$first": "$metadata.classroom_date"},
                        "count": {"$sum": 1},
                        "start": {"$min": "$timestamp"},
                        "end": {"$max": "$timestamp"},
                    }
                },
                {"$sort": {"start": 1, "inference_run_created_at": 1}},
            ]
        )
        pose_coverage_list = []
        for item in pose_coverage_cursor:
            pose_coverage_list.append(
                collections.OrderedDict(
                    (
                        ("inference_run_id", str(item["inference_run_id"])),
                        ("inference_run_created_at", item["inference_run_created_at"]),
                        ("environment_id", str(item["environment_id"])),
                        ("classroom_date", item["classroom_date"]),
                        ("count", item["count"]),
                        ("start", item["start"]),
                        ("end", item["end"]),
                    )
                )
            )
        df_pose_coverage = None
        if len(pose_coverage_list) > 0:
            df_pose_coverage = (
                pd.DataFrame(pose_coverage_list)
                .sort_values([
                    'start',
                    'inference_run_created_at'
                ])
            )
        df_pose_coverage = PoseHandle.add_inference_run_group_ids(df_pose_coverage)
        return df_pose_coverage

    @staticmethod
    def add_inference_run_group_ids(coverage_summary):
        coverage_summary = (
            coverage_summary
            .copy()
            .sort_values('start')
        )
        previous_group_id = uuid.uuid4()
        previous_end = None
        group_ids = list()
        for _, row in coverage_summary.iterrows():
            if previous_end is None or row['start'] - previous_end <= datetime.timedelta(microseconds=100000):
                group_id = previous_group_id
            else:
                group_id = uuid.uuid4()
            group_ids.append(group_id)
            previous_group_id = group_id
            previous_end = row['end']
        coverage_summary['inference_run_group_id'] = group_ids
        return coverage_summary

    def cleanup(self):
        if self.client is not None:
            self.client.close()

    def __del__(self):
        self.cleanup()
