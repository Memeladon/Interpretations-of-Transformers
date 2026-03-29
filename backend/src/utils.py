def chunk_text(text, max_length=512):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i+max_length]))

    return chunks