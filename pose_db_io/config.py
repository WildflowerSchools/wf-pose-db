import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MONGO_POSE_URI: str = "mongodb://pose-engine:iamaninsecurepassword@localhost:27017/poses?authSource=poses"
