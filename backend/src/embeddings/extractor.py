import logging
import traceback

import torch
from typing import Dict, List

from src.embeddings.aggregation import aggregate_layer

logger = logging.getLogger(__name__)


class EmbeddingExtractor:

    def __init__(self, model, tokenizer, device="cpu", max_length: int = 256):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        *,
        level: str | None = None,
        strategy: str = "mean",
    ) -> Dict:
        """
        Точка входа в модель:
        токенизация, прогон модели, затем либо возвращает сырые hidden_states (level=None),
        либо сразу агрегирует embeddings по уровню (text/sentence/token) чтобы не собирать гигантские
        тензоры формы (N, seq_len, hidden) для всех слоёв.
        """
        # Если level=None — старое поведение: вернуть hidden_states+attention_mask на все тексты.
        if level is None:
            all_hidden: list[list[torch.Tensor]] = []
            all_masks: list[torch.Tensor] = []

        n = len(texts)
        n_batches = max(1, (n + batch_size - 1) // batch_size)
        log_every = max(1, n_batches // 10)
        logger.info(
            "encode start: %s texts, batch_size=%s, max_length=%s, ~%s batches (level=%s)",
            n,
            batch_size,
            self.max_length,
            n_batches,
            level,
        )
        self.model.eval()

        # При level!=None накапливаем агрегированные представления для каждого слоя.
        layer_outputs_acc: list[list] | None = None

        for bi, start in enumerate(range(0, len(texts), batch_size)):
            batch_no = bi + 1
            batch_texts = texts[start : start + batch_size]
            chars_total = sum(len(t) for t in batch_texts)
            logger.debug(
                "encode batch %s/%s start: idx=[%s:%s], samples=%s, total_chars=%s",
                batch_no,
                n_batches,
                start,
                start + len(batch_texts),
                len(batch_texts),
                chars_total,
            )
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                )
                logger.debug(
                    "encode batch %s/%s tokenized: input_ids=%s attention_mask=%s",
                    batch_no,
                    n_batches,
                    tuple(inputs["input_ids"].shape),
                    tuple(inputs["attention_mask"].shape),
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = self.model(**inputs, output_hidden_states=True)
                logger.debug(
                    "encode batch %s/%s forward ok: n_layers=%s layer0=%s",
                    batch_no,
                    n_batches,
                    len(outputs.hidden_states),
                    tuple(outputs.hidden_states[0].shape),
                )
            except Exception as exc:
                logger.error(
                    "encode batch %s/%s failed (%s): %s",
                    batch_no,
                    n_batches,
                    type(exc).__name__,
                    exc,
                )
                logger.error("encode batch traceback:\n%s", traceback.format_exc())
                raise

            if level is None:
                if not all_hidden:
                    all_hidden = [[] for _ in range(len(outputs.hidden_states))]
                for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
                    all_hidden[layer_idx].append(layer_tensor.detach().cpu())
                all_masks.append(inputs["attention_mask"].detach().cpu())
            else:
                assert level is not None
                if layer_outputs_acc is None:
                    layer_outputs_acc = [[] for _ in range(len(outputs.hidden_states))]

                batch_mask = inputs["attention_mask"]
                # Агрегируем для каждого слоя прямо здесь — это резко уменьшает память
                # относительно хранения hidden_states формы (N, seq_len, hidden).
                for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
                    agg = aggregate_layer(
                        hidden=layer_tensor,
                        mask=batch_mask,
                        level=level,
                        tokenizer=self.tokenizer,
                        texts=batch_texts,
                        strategy=strategy,
                    )
                    if isinstance(agg, torch.Tensor):
                        layer_outputs_acc[layer_idx].append(agg.detach().cpu())
                    else:
                        # sentence/token уровни: список тензоров на каждый sample
                        layer_outputs_acc[layer_idx].extend([t.detach().cpu() for t in agg])

            # Освобождаем ссылки на outputs, чтобы Python/GC быстрее вернул память.
            del outputs

            if batch_no % log_every == 0 or batch_no == n_batches:
                logger.info("encode batch %s/%s", batch_no, n_batches)

        if level is None:
            hidden_states = tuple(torch.cat(parts, dim=0) for parts in all_hidden)
            attention_mask = torch.cat(all_masks, dim=0)
            n_layers = len(hidden_states)
            shape0 = hidden_states[0].shape
            logger.info(
                "encode done: layers=%s, first_layer_shape=%s, mask_shape=%s",
                n_layers,
                tuple(shape0),
                tuple(attention_mask.shape),
            )
            return {"hidden_states": hidden_states, "attention_mask": attention_mask}

        assert layer_outputs_acc is not None
        # Для text уровня нам нужны тензоры (N, hidden) вместо списка по батчам.
        if level == "text":
            layer_outputs = [torch.cat(parts, dim=0) for parts in layer_outputs_acc]
        else:
            layer_outputs = layer_outputs_acc

        n_layers = len(layer_outputs)
        if isinstance(layer_outputs[0], torch.Tensor):
            shape0 = layer_outputs[0].shape
            logger.info("encode aggregated done: layers=%s, first_layer_shape=%s", n_layers, tuple(shape0))
        else:
            # sentence/token: list[Tensor] (по одному элементу на sample)
            logger.info("encode aggregated done: layers=%s, first_layer_samples=%s", n_layers, len(layer_outputs[0]))
        return {"layer_outputs": layer_outputs}