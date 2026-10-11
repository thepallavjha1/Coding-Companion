"""
Codebase indexing and vector search using LangChain and Chroma.
Supports parsing full folders or individual files, chunking code,
and retrieving the most relevant chunks with file and line metadata.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .state import CodeChunk

# Common code file extensions supported for indexing
SUPPORTED_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".jsx": Language.JS,
    ".tsx": Language.TS,
    ".html": Language.HTML,
    ".css": None,
    ".json": None,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".sh": None,
    ".md": Language.MARKDOWN,
    ".sql": None,
}

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".egg-info",
    ".gemini",
    ".chroma_db",
}


class CodebaseRetriever:
    """Manages indexing and semantic vector retrieval across a codebase."""

    def __init__(
        self,
        codebase_path: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: Optional[str] = None,
    ):
        self.codebase_path = Path(codebase_path).resolve()
        self.embedding_model_name = embedding_model_name
        self.persist_dir = persist_dir
        self.vector_store: Optional[Chroma] = None
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def _collect_code_files(self) -> List[Path]:
        """Collects all code files from the target directory or file."""
        if self.codebase_path.is_file():
            return [self.codebase_path]

        collected: List[Path] = []
        for root, dirs, files in os.walk(self.codebase_path):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    collected.append(file_path)
        return collected

    def _chunk_file(self, file_path: Path) -> List[Document]:
        """Splits a single code file into manageable chunks with line number metadata."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        if not content.strip():
            return []

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        ext = file_path.suffix.lower()
        language = SUPPORTED_EXTENSIONS.get(ext)

        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=600,
                chunk_overlap=120,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=120,
            )

        text_chunks = splitter.split_text(content)
        documents: List[Document] = []

        # Calculate approximate start and end line for each chunk
        current_char_offset = 0
        for chunk in text_chunks:
            # Locate chunk position in file to calculate line numbers
            pos = content.find(chunk, current_char_offset)
            if pos == -1:
                pos = content.find(chunk)
            if pos != -1:
                start_line = content[:pos].count("\n") + 1
                end_line = start_line + chunk.count("\n")
                current_char_offset = pos + len(chunk)
            else:
                start_line = 1
                end_line = total_lines

            try:
                rel_path = str(file_path.relative_to(self.codebase_path))
            except ValueError:
                rel_path = file_path.name

            doc = Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path),
                    "relative_path": rel_path,
                    "file_name": file_path.name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_lines": total_lines,
                },
            )
            documents.append(doc)

        return documents

    def index_codebase(self) -> int:
        """Indexes all code files into Chroma vector store. Returns total chunks indexed."""
        files = self._collect_code_files()
        all_docs: List[Document] = []

        for f in files:
            file_docs = self._chunk_file(f)
            all_docs.extend(file_docs)

        if not all_docs:
            # Fallback if no files matched or empty: index empty doc
            all_docs.append(
                Document(
                    page_content="# Empty or unsupported codebase",
                    metadata={
                        "source": str(self.codebase_path),
                        "relative_path": self.codebase_path.name,
                        "start_line": 1,
                        "end_line": 1,
                    },
                )
            )

        collection_name = "codebase_index"
        self.vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_dir,
        )
        return len(all_docs)

    def search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 4,
    ) -> List[CodeChunk]:
        """
        Retrieves top relevant code chunks using semantic similarity and keyword queries.
        """
        if not self.vector_store:
            self.index_codebase()

        search_queries = [query]
        if keywords:
            search_queries.append(" ".join(keywords))

        seen_keys = set()
        results: List[CodeChunk] = []

        for q in search_queries:
            docs_with_scores = self.vector_store.similarity_search_with_score(q, k=top_k)
            for doc, score in docs_with_scores:
                meta = doc.metadata
                key = (meta.get("source"), meta.get("start_line"), meta.get("end_line"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                chunk: CodeChunk = {
                    "file_path": meta.get("source", ""),
                    "relative_path": meta.get("relative_path", ""),
                    "content": doc.page_content,
                    "start_line": int(meta.get("start_line", 1)),
                    "end_line": int(meta.get("end_line", 1)),
                    "score": float(score),
                }
                results.append(chunk)

        # Limit to top_k distinct chunks
        return results[:top_k]

    def get_full_file(self, file_path: str) -> str:
        """Reads the entire content of a file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"# Error reading file {file_path}: {e}"
