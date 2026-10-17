from pathlib import Path
from document_parser import create_langchain_documents
#Qdrant Import
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from model import load_embedding_model

#Loading Models
embeddings = load_embedding_model()
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")



def create_vectorstore(pdfs, EXTRACTED_VISUAL_PATH, progress_callback=None):
    """Build one vector store from several PDFs.

    pdfs: list of (pdf_path, display_name) tuples.
    """
    all_docs = []
    for i, (pdf_path, name) in enumerate(pdfs):
        if progress_callback:
            progress_callback(i, len(pdfs), name)

        # Per-document folder so extracted chart names don't collide
        visual_dir = Path(EXTRACTED_VISUAL_PATH) / f"doc_{i + 1}"
        visual_dir.mkdir(parents=True, exist_ok=True)

        docs = create_langchain_documents(pdf_path, visual_dir)
        for d in docs:
            d.metadata["source_file"] = name
            d.metadata["chunk_id"] = f"doc{i + 1}_{d.metadata.get('chunk_id')}"
        all_docs.extend(docs)

    client = QdrantClient(path=":memory:")

    # Create a collection with both dense and sparse vectors
    client.create_collection(
        collection_name="pdf_collection",
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )

    qdrant = QdrantVectorStore(
        client=client,
        collection_name="pdf_collection",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    qdrant.add_documents(all_docs)
    return qdrant