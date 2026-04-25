import nest_asyncio
nest_asyncio.apply()
import asyncio
import boto3
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import Document, Settings
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

app = FastAPI(title="RAG Grafo - Legal Peruano", version="3.0")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL  = "http://localhost:11434"

# Configuración MinIO
MINIO_ENDPOINT   = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME      = "salpma1"

# Configuración Neo4j
NEO4J_URL        = "bolt://localhost:7687"
NEO4J_USER       = "neo4j"
NEO4J_PASS       = "password123"

# Modelos
DEFAULT_LLM_MODEL   = "llama3.1"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------------
# INICIALIZACIÓN
# ---------------------------------------------------------------------------
class PreguntaDTO(BaseModel):
    pregunta: str

class TextoDTO(BaseModel):
    texto: str

Settings.llm = Ollama(
    model=DEFAULT_LLM_MODEL,
    temperature=0.0,
    request_timeout=300.0,
    base_url=OLLAMA_BASE_URL
)
Settings.embed_model = OllamaEmbedding(
    model_name=DEFAULT_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

# Variable global para el índice y executor para threads
graph_index = None
executor = ThreadPoolExecutor(max_workers=1)

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )

def init_graph_store():
    return Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASS,
        url=NEO4J_URL,
    )

# Intentar cargar índice existente al arrancar
try:
    _store = init_graph_store()
    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=_store,
        embed_model=Settings.embed_model,
    )
    print("✅ Índice de grafos cargado desde Neo4j.")
except Exception as e:
    print(f"⚠️ Motor de grafos en espera de /cargar o /cargartexto: {e}")

# ---------------------------------------------------------------------------
# FUNCIÓN COMPARTIDA DE CONSTRUCCIÓN DE GRAFO (corre en thread separado)
# ---------------------------------------------------------------------------
def _build_graph(documentos: list) -> PropertyGraphIndex:
    """
    Construye el grafo de conocimiento a partir de una lista de Documents.
    Se ejecuta en un thread separado para evitar conflictos de event loop.
    """
    graph_store = init_graph_store()

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=64,
    )

    extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=5,
        num_workers=1,
    )

    idx = PropertyGraphIndex.from_documents(
        documentos,
        property_graph_store=graph_store,
        kg_extractors=[extractor],
        transformations=[splitter],
        embed_model=Settings.embed_model,
        show_progress=True,
    )
    return idx

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/cargartexto")
async def cargar_texto(data: TextoDTO):
    """Recibe texto plano en JSON y lo indexa en el grafo de conocimiento."""
    global graph_index

    if not data.texto.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")

    try:
        print(f"📝 Texto recibido ({len(data.texto)} chars). Iniciando extracción...")

        documento = Document(
            text=data.texto,
            metadata={"fuente": "input_manual"}
        )

        loop = asyncio.get_running_loop()
        graph_index = await loop.run_in_executor(executor, _build_graph, [documento])

        return {
            "mensaje": f"Grafo construido exitosamente.",
            "caracteres": len(data.texto)
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cargar")
async def cargar_desde_minio():
    """Lee todos los archivos del bucket MinIO y los indexa en el grafo."""
    global graph_index

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
                documentos.append(Document(
                    text=texto,
                    metadata={"fuente": key}
                ))

        if not documentos:
            raise HTTPException(status_code=400, detail="No se encontró texto válido en el bucket.")

        print(f"🧠 Iniciando extracción de Grafos con {len(documentos)} documento(s)...")

        loop = asyncio.get_running_loop()
        graph_index = await loop.run_in_executor(executor, _build_graph, [documento])

        return {
            "mensaje": "Grafo de conocimiento construido exitosamente en Neo4j.",
            "documentos_procesados": len(documentos)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preguntar")
async def preguntar(data: PreguntaDTO):
    """Consulta el grafo de conocimiento con una pregunta en lenguaje natural."""
    global graph_index

    if not data.pregunta.strip():
        raise HTTPException(status_code=400, detail="Pregunta vacía.")
    if graph_index is None:
        raise HTTPException(status_code=400, detail="Ejecuta /cargar o /cargartexto primero.")

    try:
        query_engine = graph_index.as_query_engine(
            include_text=True,
            similarity_top_k=5
        )
        respuesta = query_engine.query(data.pregunta)
        return {
            "pregunta": data.pregunta,
            "respuesta": str(respuesta)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/estado")
def estado():
    """Verifica si el índice está cargado y listo para consultas."""
    return {
        "indice_cargado": graph_index is not None,
        "listo_para_consultas": graph_index is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)