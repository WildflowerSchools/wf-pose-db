from typing import List

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

    def cleanup(self):
        if self.client is not None:
            self.client.close()

    def __del__(self):
        self.cleanup()
