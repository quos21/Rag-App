import os
import json
import faiss
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from nanoid import generate
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class RAGManager:
    def __init__(self):
        self.index_path = Path("embeddings/index.faiss")
        self.metadata_path = Path("metadata/documents.json")

        self.index_path.parent.mkdir(exist_ok=True)
        self.metadata_path.parent.mkdir(exist_ok=True)

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY2"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION2", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT2"),
        )

        self.embedding_deployment = os.getenv(
            "AZURE_EMBEDDING_DEPLOYMENT2", "text-embedding-3-large"
        )
        self.embedding_dim = 3072

        self.index = self._load_index()
        self.metadata = self._load_metadata()

    def _load_index(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatL2(self.embedding_dim)

    def _load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {"documents": {}, "chunks": []}

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _get_embedding(self, text: str) -> np.ndarray:
        emb = self.client.embeddings.create(
            model=self.embedding_deployment, input=text
        )
        return np.array(emb.data[0].embedding, dtype=np.float32)

    def add_document(self, pdf_path: str, doc_id: str | None = None) -> Dict:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        words = text.split()
        chunks = [
            " ".join(words[i : i + 500]) for i in range(0, len(words), 450)
        ]

        doc_id = doc_id or generate(size=8)
        embeddings = [self._get_embedding(c) for c in chunks]

        start_idx = len(self.metadata["chunks"])
        self.index.add(np.array(embeddings))

        chunk_ids = list(range(start_idx, start_idx + len(chunks)))
        self.metadata["documents"][doc_id] = {
            "filename": Path(pdf_path).name,
            "uploaded_at": datetime.now().isoformat(),
            "chunk_ids": chunk_ids,
        }

        for i, chunk in enumerate(chunks):
            self.metadata["chunks"].append(
                {"doc_id": doc_id, "chunk_id": start_idx + i, "text": chunk}
            )

        self._save()
        return {"doc_id": doc_id, "chunks": len(chunks)}

    def delete_document(self, doc_id: str) -> Dict:
        if doc_id not in self.metadata["documents"]:
            raise ValueError("Document not found")

        self.metadata["chunks"] = [
            c for c in self.metadata["chunks"] if c["doc_id"] != doc_id
        ]
        del self.metadata["documents"][doc_id]

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if self.metadata["chunks"]:
            embeddings = [
                self._get_embedding(c["text"]) for c in self.metadata["chunks"]
            ]
            self.index.add(np.array(embeddings))
            for i, c in enumerate(self.metadata["chunks"]):
                c["chunk_id"] = i

        self._save()
        return {"deleted": doc_id}

    def list_documents(self) -> List[Dict]:
        return [
            {"doc_id": k, **v} for k, v in self.metadata["documents"].items()
        ]

    def search(self, query: str, top_k: int = 3) -> Dict:
        if self.index.ntotal == 0:
            return {"found": False, "results": []}

        q_emb = self._get_embedding(query)
        dists, idxs = self.index.search(
            np.array([q_emb]), min(top_k, self.index.ntotal)
        )

        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            chunk = self.metadata["chunks"][idx]
            doc = self.metadata["documents"][chunk["doc_id"]]
            results.append(
                {
                    "text": chunk["text"],
                    "source": doc["filename"],
                    "score": float(dist),
                }
            )

        return {"found": True, "results": results}
