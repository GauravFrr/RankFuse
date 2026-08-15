def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of character size with overlap.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum character count per chunk.
        overlap: Character overlap between adjacent chunks.

    Returns:
        A list of text chunk strings.
    """
    if not text:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    # Safety check to prevent infinite loop
    if step <= 0:
        step = chunk_size

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
