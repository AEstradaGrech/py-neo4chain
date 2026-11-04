from pydantic import BaseModel, Field
from api.infrastructure.llm.statics import default_embedding_model, default_embedding_dimensions, default_embedding_numctx
class OllamaEmbeddingsConfig(BaseModel):
    model:str = Field(description="Embeddings model to use", default=default_embedding_model)
    dimensions:int = Field(description="Dimensions of the embedding", default=default_embedding_dimensions)
    num_ctx:str = Field(description="Context window size for the embedder", default=default_embedding_numctx)