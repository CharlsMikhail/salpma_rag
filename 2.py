import boto3
from botocore.config import Config
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import qdrant_client

app = FastAPI(title="RAG Avanzado - Sentence Window & Nomic", version="2.0")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "documentos_salpma2" # Nueva colección para no chocar
OLLAMA_BASE_URL  = "http://localhost:11434"

# Configuración MinIO
MINIO_ENDPOINT   = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME      = "salpma1"

# Modelos y RAG
DEFAULT_LLM_MODEL        = "llama3.1"
DEFAULT_EMBED_MODEL      = "nomic-embed-text" # <-- Nuevo modelo de embedding
DEFAULT_EMBED_DIMS       = 768                # <-- Dimensiones para Nomic
DEFAULT_RERANKER_MODEL   = "BAAI/bge-reranker-base" # <-- Descarga directa de HF
DEFAULT_RETRIEVAL_TOP_K  = 15  # Buscamos más oraciones exactas
DEFAULT_SIMILARITY_TOP_K = 5   # Filtramos a los mejores 5 contextos expandidos

RAG_SYSTEM_PROMPT = """Eres un asistente legal experto. Responde ÚNICAMENTE basándote \
en el contexto proporcionado. Cita la información con precisión.
Si la información NO está, responde: "No encontré información suficiente."
"""
RAG_QA_TEMPLATE = PromptTemplate(RAG_SYSTEM_PROMPT + "\nPregunta: {query_str}\nRespuesta:")

# ---------------------------------------------------------------------------
# INICIALIZACIÓN DE MOTOR Y ESTADO GLOBAL
# ---------------------------------------------------------------------------
class PreguntaDTO(BaseModel):
    pregunta: str

Settings.llm = Ollama(model=DEFAULT_LLM_MODEL, temperature=0.1, request_timeout=120.0, base_url=OLLAMA_BASE_URL)
Settings.embed_model = OllamaEmbedding(model_name=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

query_engine = None

def build_query_engine(index: VectorStoreIndex):
    """Construye el motor con inyección de contexto (Window) y Reranker PyTorch."""
    
    # 1. Postprocesador para reemplazar la oración corta por el párrafo completo
    metadata_replacement = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )
    
    # 2. Reranker Nativo (se descarga a la caché de HuggingFace en el primer uso)
    reranker = SentenceTransformerRerank(
        model=DEFAULT_RERANKER_MODEL, 
        top_n=DEFAULT_SIMILARITY_TOP_K
    )
    
    return index.as_query_engine(
        similarity_top_k=DEFAULT_RETRIEVAL_TOP_K,
        text_qa_template=RAG_QA_TEMPLATE,
        node_postprocessors=[metadata_replacement, reranker],
    )

def get_s3_client():
    return boto3.client(
        "s3", endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY, region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )

try:
    _client = qdrant_client.QdrantClient(url=QDRANT_URL)
    if _client.collection_exists(COLLECTION_NAME):
        _vector_store = QdrantVectorStore(client=_client, collection_name=COLLECTION_NAME)
        _index = VectorStoreIndex.from_vector_store(_vector_store)
        query_engine = build_query_engine(_index)
except Exception as e:
    print(f"⚠️ Motor base en espera de /cargar: {e}")

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/cargar")
def cargar_desde_minio():
    global query_engine
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        objetos = response.get("Contents", [])

        if not objetos:
            raise HTTPException(status_code=400, detail="Bucket vacío.")

        documentos = []
        for obj in objetos:
            key = obj["Key"]
            print(f"📄 Leyendo: {key}")
            body = s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()
            texto = body.decode("utf-8", errors="ignore")
            if texto.strip():
                documentos.append(Document(text=texto, metadata={"fuente": key}))

        # --- MAGIA DEL SENTENCE WINDOW PARSER ---
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3, # Inyecta 3 oraciones antes y 3 después
            window_metadata_key="window",
            original_text_metadata_key="original_sentence",
        )
        nodes = node_parser.get_nodes_from_documents(documentos)
        print(f"🔪 Textos divididos en {len(nodes)} nodos (oraciones) con ventanas de contexto.")

        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_client.http.models.VectorParams(
                size=DEFAULT_EMBED_DIMS, distance="Cosine"
            ),
        )

        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Indexamos los NODOS, no los documentos crudos
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        query_engine = build_query_engine(index)

        return {"mensaje": "Indexación Avanzada exitosa", "nodos_oracion_creados": len(nodes)}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preguntar")
def preguntar(data: PreguntaDTO):
    global query_engine
    if not data.pregunta.strip():
        raise HTTPException(status_code=400, detail="Pregunta vacía.")
    if query_engine is None:
        raise HTTPException(status_code=400, detail="Ejecuta /cargar primero.")
    
    try:
        respuesta = query_engine.query(data.pregunta)
        return {"pregunta": data.pregunta, "respuesta": str(respuesta)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) # Puerto 8001 para no chocar con el Stack 1