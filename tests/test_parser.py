"""
Tests for src/parser.py

clean_text and chunk_text are pure functions — no mocking required.
load_documents tests use reportlab to write PDFs with real extractable text,
since pypdf's PdfWriter produces blank pages with no extractable content.
"""

import pytest
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.parser import clean_text, chunk_text, load_documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_text_pdf(path: Path, content: str) -> None:
    """
    Write a PDF with real extractable text using reportlab.
    pypdf PdfWriter.add_blank_page() produces pages with no text layer,
    so extract_text() returns '' — useless for testing load_documents.
    """
    from reportlab.pdfgen import canvas as rl_canvas

    c = rl_canvas.Canvas(str(path))
    # Wrap content across lines at ~80 chars to stay within page margins.
    line_height = 14
    y = 750
    for i in range(0, len(content), 80):
        if y < 50:
            c.showPage()
            y = 750
        c.drawString(72, y, content[i:i + 80])
        y -= line_height
    c.save()


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_collapses_multiple_spaces(self):
        assert clean_text("too   many    spaces") == "too many spaces"

    def test_collapses_newlines(self):
        assert clean_text("line one\n\nline two") == "line one line two"

    def test_collapses_tabs(self):
        assert clean_text("col1\t\tcol2") == "col1 col2"

    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_text("   leading and trailing   ") == "leading and trailing"

    def test_mixed_whitespace_artefacts(self):
        """Typical PDF extraction output: mixed newlines, tabs, and runs."""
        dirty = "This   has \n\n weird \t whitespace."
        assert clean_text(dirty) == "This has weird whitespace."

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_only_whitespace_returns_empty(self):
        assert clean_text("   \n\t  ") == ""

    def test_already_clean_text_is_unchanged(self):
        text = "This sentence is already clean."
        assert clean_text(text) == text

    def test_single_internal_space_preserved(self):
        """Must not strip intentional single spaces between words."""
        assert clean_text("word1 word2") == "word1 word2"

    def test_preserves_punctuation(self):
        assert clean_text("Hello, world!") == "Hello, world!"


# ---------------------------------------------------------------------------
# chunk_text — overlap math and data integrity
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_first_chunk_is_correct(self):
        chunks = chunk_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chunk_size=10, overlap=5)
        assert chunks[0] == "ABCDEFGHIJ"

    def test_second_chunk_applies_overlap_correctly(self):
        """
        step = chunk_size - overlap = 10 - 5 = 5.
        Chunk 2 starts at index 5.
        """
        chunks = chunk_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chunk_size=10, overlap=5)
        assert chunks[1] == "FGHIJKLMNO"

    def test_no_characters_lost_at_tail(self):
        """The last character must appear somewhere in the output chunks."""
        chunks = chunk_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chunk_size=10, overlap=5)
        assert "".join(chunks).endswith("Z")

    def test_every_character_covered_at_least_once(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, overlap=5)
        covered = set("".join(chunks))
        assert set(text).issubset(covered)

    def test_full_chunks_do_not_exceed_chunk_size(self):
        """Every chunk except possibly the last must be exactly chunk_size characters."""
        chunks = chunk_text("A" * 100, chunk_size=20, overlap=5)
        for chunk in chunks[:-1]:
            assert len(chunk) == 20

    def test_overlap_content_matches_for_full_chunks(self):
        """
        For adjacent pairs where both chunks are full-length, the last `overlap`
        chars of chunk N must equal the first `overlap` chars of chunk N+1.
        The tail chunk may be shorter than the overlap window — skip it.
        """
        overlap = 5
        chunks = chunk_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chunk_size=10, overlap=overlap)
        for i in range(len(chunks) - 1):
            # Only assert when the right chunk is long enough to contain the overlap.
            if len(chunks[i + 1]) >= overlap:
                assert chunks[i][-overlap:] == chunks[i + 1][:overlap]

    def test_no_overlap_produces_contiguous_non_overlapping_chunks(self):
        chunks = chunk_text("ABCDEFGHIJ", chunk_size=5, overlap=0)
        assert chunks == ["ABCDE", "FGHIJ"]

    def test_text_shorter_than_chunk_size_returns_single_chunk(self):
        chunks = chunk_text("ABC", chunk_size=10, overlap=2)
        assert chunks == ["ABC"]

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", chunk_size=10, overlap=2) == []

    def test_exact_chunk_size_match_returns_single_chunk(self):
        chunks = chunk_text("ABCDE", chunk_size=5, overlap=0)
        assert chunks == ["ABCDE"]

    def test_step_determines_chunk_count(self):
        """
        26 chars, chunk_size=10, overlap=5 → step=5.
        Starts at: 0, 5, 10, 15, 20, 25 → 6 chunks.
        (The tail chunk at index 25 is just 'Z'.)
        """
        chunks = chunk_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chunk_size=10, overlap=5)
        assert len(chunks) == 6


# ---------------------------------------------------------------------------
# load_documents — filesystem integration
# ---------------------------------------------------------------------------

class TestLoadDocuments:
    def test_raises_file_not_found_on_missing_directory(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_documents(missing, chunk_size=500, overlap=50)

    def test_returns_empty_list_on_empty_directory(self, tmp_path):
        result = load_documents(tmp_path, chunk_size=500, overlap=50)
        assert result == []

    def test_record_schema_has_required_keys(self, tmp_path):
        """
        Each record must have source, chunk_index, and text —
        these are the exact fields embedder.build_collection reads.
        """
        _write_text_pdf(tmp_path / "test.pdf", "Extractable content. " * 50)
        records = load_documents(tmp_path, chunk_size=100, overlap=10)
        assert len(records) > 0
        for record in records:
            assert "source" in record
            assert "chunk_index" in record
            assert "text" in record

    def test_chunk_indices_are_sequential_from_zero(self, tmp_path):
        _write_text_pdf(tmp_path / "doc.pdf", "A" * 500)
        records = load_documents(tmp_path, chunk_size=100, overlap=10)
        indices = [r["chunk_index"] for r in records]
        assert indices == list(range(len(indices)))

    def test_source_field_is_filename_not_full_path(self, tmp_path):
        _write_text_pdf(tmp_path / "annual_report.pdf", "Content. " * 50)
        records = load_documents(tmp_path, chunk_size=500, overlap=50)
        assert all(r["source"] == "annual_report.pdf" for r in records)

    def test_multiple_pdfs_produce_records_for_each(self, tmp_path):
        _write_text_pdf(tmp_path / "doc_a.pdf", "Alpha content. " * 30)
        _write_text_pdf(tmp_path / "doc_b.pdf", "Beta content. " * 30)
        records = load_documents(tmp_path, chunk_size=200, overlap=20)
        sources = {r["source"] for r in records}
        assert sources == {"doc_a.pdf", "doc_b.pdf"}

    def test_non_pdf_files_are_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("This should be ignored.")
        (tmp_path / "data.csv").write_text("a,b,c")
        result = load_documents(tmp_path, chunk_size=500, overlap=50)
        assert result == []

    def test_text_field_is_non_empty_string(self, tmp_path):
        _write_text_pdf(tmp_path / "doc.pdf", "The quick brown fox. " * 20)
        records = load_documents(tmp_path, chunk_size=100, overlap=10)
        for record in records:
            assert isinstance(record["text"], str)
            assert len(record["text"]) > 0