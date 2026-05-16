from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    debug: bool = Field(default=False)
    
    # Context Limits 
    sql_max_rows: int = Field(default=50, description="Max rows in SqlResult")
    evidence_max_art: int = Field(default=6, description="Max articles in EvidenceBundle")
    evidence_retrieval_pool_max: int = Field(default=200, description="PubMed pre-rerank pool size")
    article_max_snippet: int = Field(default=500, description="Chars of abstract per article")
    clinical_max_list: int = Field(default=10, description="Items per list in ClinicalContext")
    
    # LLM Settings
    copilot_llm_profile: str = Field(default="custom")
    copilot_query_planner: str = Field(default="heuristic")
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Synthea SQL DB
    synthea_db_path: str = Field(default="data/clinical/synthea.db")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()
