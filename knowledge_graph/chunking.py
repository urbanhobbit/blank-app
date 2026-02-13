"""Text chunking utilities for splitting documents into processable segments."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split text into word-based chunks with overlap.

    Args:
        text: The input text to chunk.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
