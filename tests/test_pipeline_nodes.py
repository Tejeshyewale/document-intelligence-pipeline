"""
Basic unit tests for pipeline nodes that don't require downloading
large transformer models (fast, deterministic, CI-friendly).
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

from nodes.text_chunker.code import TextChunker
from nodes.document_loader.code import DocumentLoader


def test_chunker_produces_chunks():
    chunker = TextChunker({'chunk_size': 5, 'overlap': 1})
    text = "one two three four five six seven eight nine ten"
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunker_empty_text():
    chunker = TextChunker({'chunk_size': 5, 'overlap': 1})
    chunks = chunker.chunk("")
    assert chunks == []


def test_document_loader_missing_file():
    loader = DocumentLoader({})
    with pytest.raises(FileNotFoundError):
        loader.load("does_not_exist.txt")


def test_document_loader_unsupported_type(tmp_path):
    loader = DocumentLoader({})
    bad_file = tmp_path / "file.docx"
    bad_file.write_text("hello")
    with pytest.raises(ValueError):
        loader.load(str(bad_file))


def test_document_loader_empty_txt(tmp_path):
    loader = DocumentLoader({})
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   ")
    with pytest.raises(ValueError):
        loader.load(str(empty_file))


def test_document_loader_reads_txt(tmp_path):
    loader = DocumentLoader({})
    f = tmp_path / "sample.txt"
    f.write_text("Hello world")
    assert loader.load(str(f)) == "Hello world"
