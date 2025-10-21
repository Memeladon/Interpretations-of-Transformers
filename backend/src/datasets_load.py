import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# === Semantic tasks ===
ruSTS1 = load_dataset("ai-forever/ru-stsbenchmark-sts") # <sentence1, score[0;5], sentence2>
ruSTS2 = load_dataset("Alexator26/eval-ru-sts-dataset") # <text1, text2, label[0;5]>
ruQQP = load_dataset("MilyaShams/qqp-ru_10k") #  <text1, text2, label(0,1)>
ruSNLI = load_dataset("MilyaShams/snli-ru_10k") # <premise, hypothesis, label(0-следствие, 1-нейтральный, 2-противоречие)>

# === Style tasks ===
ruSentiment = load_dataset("MonoHime/ru_sentiment_dataset") # <Unnamed, text, sentiment(0-нейтр, 1-поз, 2-нег)>
ruEmotions = load_dataset("seara/ru_go_emotions", "raw") #

unified_splits = []

# ---------- SEMANTIC ----------
def format_semantic(ds, text1, text2, label, label_type="regression"):
    return ds.map(lambda x: {
        "text1": x[text1],
        "text2": x[text2],
        "label": float(x[label]),
        "task_type": "semantic",
        "label_type": label_type
    }, remove_columns=ds.column_names)

unified_splits += [
    format_semantic(ruSTS1["test"], "sentence1", "sentence2", "score"),
    format_semantic(ruSTS2["test"], "text1", "text2", "label"),
    format_semantic(ruQQP["train"], "text1", "text2", "label", label_type="classification"),
    format_semantic(ruSNLI["train"], "premise", "hypothesis", "label", label_type="classification")
]

# ---------- STYLE ----------
def format_style(ds, text, label, label_type="classification"):
    return ds.map(lambda x: {
        "text1": x[text],
        "text2": None,
        "label": int(x[label]),
        "task_type": "style",
        "label_type": label_type
    }, remove_columns=ds.column_names)

unified_splits += [
    format_style(ruSentiment["train"], "text", "sentiment"),
    format_style(ruEmotions["train"], "text", "labels")
]

# ---------- Combine all ----------
unified_dataset = concatenate_datasets(unified_splits)
return unified_dataset
