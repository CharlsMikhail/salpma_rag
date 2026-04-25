import re
import httpx
import boto3
from botocore.config import Config
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import qdrant_client

app = FastAPI(title="RAG API - Stack Base", version="1.1")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "documentos_salpma1"
OLLAMA_BASE_URL  = "http://localhost:11434"

# Configuración MinIO
MINIO_ENDPOINT   = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME      = "salpma1"

# Modelos y RAG
DEFAULT_LLM_MODEL        = "llama3.1"
DEFAULT_EMBED_MODEL      = "bge-m3"
DEFAULT_EMBED_DIMS       = 1024
DEFAULT_RERANKER_MODEL   = "dengcao/bge-reranker-v2-m3"
DEFAULT_RETRIEVAL_TOP_K  = 10  
DEFAULT_SIMILARITY_TOP_K = 5   

RAG_SYSTEM_PROMPT = """Eres un asistente legal especializado. Responde ÚNICAMENTE basándose \
en el contexto proporcionado por los documentos indexados.
Si la información NO está, responde: "No encontré información suficiente."
"""
RAG_QA_TEMPLATE = PromptTemplate(RAG_SYSTEM_PROMPT + "\nPregunta: {query_str}\nRespuesta:")

# ---------------------------------------------------------------------------
# RERANKER (Tu lógica híbrida)
# ---------------------------------------------------------------------------
class OllamaBGEReranker(BaseNodePostprocessor):
    model: str    = DEFAULT_RERANKER_MODEL
    base_url: str = OLLAMA_BASE_URL
    top_n: int    = DEFAULT_SIMILARITY_TOP_K

    class Config:
        arbitrary_types_allowed = True

    def _rerank_native(self, query: str, documents: List[str]) -> Optional[List[float]]:
        payload = {"model": self.model, "query": query, "documents": documents}
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(f"{self.base_url}/api/rerank", json=payload)
                if resp.status_code == 404: return None
                resp.raise_for_status()
                results = resp.json().get("results", [])
                scores  = [0.0] * len(documents)
                for r in results:
                    idx = r.get("index", -1)
                    if 0 <= idx < len(scores):
                        scores[idx] = float(r.get("relevance_score", 0.0))
                return scores
        except Exception:
            return None

    def _score_one(self, client: httpx.Client, query: str, document: str) -> float:
        prompt = (
            "Given a QUERY and a DOCUMENT, output ONLY a single decimal number "
            "between 0.0 and 1.0 indicating relevance.\n\n"
            f"QUERY: {query}\nDOCUMENT: {document[:600]}\nSCORE:"
        )
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_predict": 8},
        }
        try:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response", "0").strip()
            match = re.search(r"\d+\.?\d*", raw)
            score = float(match.group()) if match else 0.0
            return min(score, 1.0) if score <= 1.0 else score / 100.0
        except Exception:
            return 0.0

    def _rerank_fallback(self, query: str, documents: List[str]) -> List[float]:
        scores = []
        with httpx.Client(timeout=90.0) as client:
            for doc in documents:
                scores.append(self._score_one(client, query, doc))
        return scores

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if not nodes or query_bundle is None: return nodes[:self.top_n]
        query = query_bundle.query_str
        doc_texts = [n.get_content() for n in nodes]
        scores = self._rerank_native(query, doc_texts)
        if scores is None: scores = self._rerank_fallback(query, doc_texts)
        for node, score in zip(nodes, scores): node.score = score
        reranked = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
        return reranked[:self.top_n]

# ---------------------------------------------------------------------------
# INICIALIZACIÓN DE MOTOR Y ESTADO GLOBAL
# ---------------------------------------------------------------------------
class PreguntaDTO(BaseModel):
    pregunta: str

Settings.llm = Ollama(model=DEFAULT_LLM_MODEL, temperature=0.1, request_timeout=120.0, base_url=OLLAMA_BASE_URL)
Settings.embed_model = OllamaEmbedding(model_name=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

query_engine = None

def build_query_engine(index: VectorStoreIndex):
    reranker = OllamaBGEReranker(model=DEFAULT_RERANKER_MODEL, base_url=OLLAMA_BASE_URL, top_n=DEFAULT_SIMILARITY_TOP_K)
    return index.as_query_engine(
        similarity_top_k=DEFAULT_RETRIEVAL_TOP_K,
        text_qa_template=RAG_QA_TEMPLATE,
        node_postprocessors=[reranker],
    )

def get_s3_client():
    return boto3.client(
        "s3", endpoint_url=MINIO_ENDPOINT, aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY, region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )

# Intenta cargar el motor si Qdrant ya tiene datos al iniciar
try:
    _client = qdrant_client.QdrantClient(url=QDRANT_URL)
    if _client.collection_exists(COLLECTION_NAME):
        _vector_store = QdrantVectorStore(client=_client, collection_name=COLLECTION_NAME)
        _index = VectorStoreIndex.from_vector_store(_vector_store)
        query_engine = build_query_engine(_index)
except Exception as e:
    print(f"⚠️ No se pudo inicializar el motor base al arrancar: {e}")

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
            raise HTTPException(status_code=400, detail=f"El bucket '{BUCKET_NAME}' está vacío.")

        documentos = []
        for obj in objetos:
            key = obj["Key"]
            print(f"📄 Leyendo desde MinIO: {key}")
            body = s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()
            texto = body.decode("utf-8", errors="ignore")
            if texto.strip():
                documentos.append(Document(text=texto, metadata={"fuente": key}))

        if not documentos:
            raise HTTPException(status_code=400, detail="Archivos sin contenido legible.")

        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        
        # Validar o crear colección en Qdrant
        if client.collection_exists(COLLECTION_NAME):
            info = client.get_collection(COLLECTION_NAME)
            stored_dims = info.config.params.vectors.size
            if stored_dims != DEFAULT_EMBED_DIMS:
                print("⚠️ Dimensiones incompatibles. Recreando colección…")
                client.delete_collection(COLLECTION_NAME)

        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=DEFAULT_EMBED_DIMS,
                    distance="Cosine",
                ),
            )

        # Indexación
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documentos, storage_context=storage_context)
        
        # Actualizar el motor global para que /preguntar funcione de inmediato
        query_engine = build_query_engine(index)

        return {
            "mensaje": "Carga y vectorización exitosa",
            "docs_indexados": len(documentos)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR en /cargar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preguntar")
def preguntar(data: PreguntaDTO):
    global query_engine
    if not data.pregunta.strip():
        raise HTTPException(status_code=400, detail="Pregunta vacía.")
    if query_engine is None:
        raise HTTPException(status_code=400, detail="El motor no está listo. Ejecuta /cargar primero.")
    
    try:
        respuesta = query_engine.query(data.pregunta)
        return {"pregunta": data.pregunta, "respuesta": str(respuesta)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)