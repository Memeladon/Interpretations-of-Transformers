import torch
import re


def mean_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def cls_pooling(hidden: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0]


def token_level(hidden: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    results: list[torch.Tensor] = []
    for idx in range(hidden.shape[0]):
        seq_len = int(mask[idx].sum().item())
        results.append(hidden[idx, :seq_len])
    return results


def text_level(hidden: torch.Tensor, mask: torch.Tensor, strategy: str = "mean") -> torch.Tensor:
    if strategy == "mean":
        return mean_pooling(hidden, mask)
    if strategy == "cls":
        return cls_pooling(hidden)
    raise ValueError(f"Unknown text-level strategy: {strategy}")


def _split_sentences(text: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return chunks if chunks else [text]


def sentence_level(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
    texts: list[str],
) -> list[torch.Tensor]:
    """
    Агрегация на уровне предложений:
    делим текст на предложения и последовательно соотносим их с токенами.
    """
    results: list[torch.Tensor] = []
    for i, text in enumerate(texts):
        valid_len = int(mask[i].sum().item())
        token_embeds = hidden[i, 1 : max(2, valid_len - 1)]  # без спецтокенов
        sentences = _split_sentences(text)

        sent_sizes = []
        for sent in sentences:
            token_ids = tokenizer.encode(sent, add_special_tokens=False)
            sent_sizes.append(max(1, len(token_ids)))

        if not sent_sizes or token_embeds.shape[0] == 0:
            results.append(hidden[i, :1])
            continue

        total = sum(sent_sizes)
        cursor = 0
        vectors: list[torch.Tensor] = []
        fallback_vec = token_embeds.mean(dim=0)
        for size in sent_sizes:
            scaled = max(1, int(round(size * token_embeds.shape[0] / total)))
            end = min(token_embeds.shape[0], cursor + scaled)
            if end <= cursor:
                end = min(token_embeds.shape[0], cursor + 1)
            segment = token_embeds[cursor:end]
            if segment.shape[0] == 0:
                vectors.append(fallback_vec)
            else:
                vectors.append(segment.mean(dim=0))
            cursor = end
        results.append(torch.stack(vectors))
    return results


def token_level_raw(hidden):
    return hidden


def aggregate_layer(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    level: str,
    tokenizer=None,
    texts: list[str] | None = None,
    strategy: str = "mean",
):
    if level == "token":
        return token_level(hidden, mask)
    if level == "text":
        return text_level(hidden, mask, strategy=strategy)
    if level == "sentence":
        if tokenizer is None or texts is None:
            raise ValueError("sentence-level aggregation needs tokenizer and texts")
        return sentence_level(hidden, mask, tokenizer, texts)
    raise ValueError(f"Unknown level: {level}")