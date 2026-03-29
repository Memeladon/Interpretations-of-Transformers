Датасеты должны быть унифицированы. Схема записи в датасете:
{
    "id": str,
    "text": str OR List[str],
    "text_pair": Optional[str],
    "label": float | int,
    "task_type": "classification" | "regression",
    "task_name": str
}