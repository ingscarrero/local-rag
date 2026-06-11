"""Unit tests for text chunking — no model servers required."""

from local_rag.ingest.chunk import chunk_page_text


def test_empty_text_returns_empty():
    assert chunk_page_text("", page_number=1) == []


def test_whitespace_only_returns_empty():
    assert chunk_page_text("   \n\t  ", page_number=1) == []


def test_single_chunk_short_text():
    text = "word " * 10
    chunks = chunk_page_text(text.strip(), page_number=1)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_multiple_chunks_produced():
    # 500 words → should produce more than one chunk with default settings (220 words)
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_page_text(text, page_number=2)
    assert len(chunks) > 1


def test_overlap_means_words_appear_in_consecutive_chunks():
    """With overlap=2, the last 2 words of chunk N should appear in chunk N+1."""
    text = " ".join(str(i) for i in range(20))
    chunks = chunk_page_text(text, page_number=1, words_per_chunk=5, overlap=2)
    assert len(chunks) >= 2
    tail = chunks[0].text.split()[-2:]
    head = chunks[1].text.split()[:2]
    assert tail == head, f"Expected overlap: {tail} != {head}"


def test_page_number_preserved():
    text = " ".join(f"w{i}" for i in range(300))
    for page in (1, 5, 99):
        chunks = chunk_page_text(text, page_number=page)
        assert all(c.page_number == page for c in chunks)


def test_start_index_offset():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_page_text(text, page_number=1, start_index=10)
    assert chunks[0].chunk_index == 10
    assert chunks[-1].chunk_index == 10 + len(chunks) - 1


def test_chunk_indices_are_contiguous():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_page_text(text, page_number=1, start_index=0)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_text_is_non_empty():
    text = " ".join(f"w{i}" for i in range(500))
    for c in chunk_page_text(text, page_number=1):
        assert c.text.strip(), "Chunk text must not be empty"


def test_all_words_covered():
    """Every word in the source must appear in at least one chunk."""
    words = [f"unique{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_page_text(text, page_number=1, words_per_chunk=10, overlap=2)
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(words)
