"""Compatibilidad hacia helpers de persistencia en core.persistence."""

from core.persistence.persistence_flow_helpers import (
    _jsonl_dict_to_msg,
    _load_jsonl,
    _msg_to_jsonl_dict,
    _role_from_msg,
    _row_to_msg,
    _save_jsonl,
)

__all__ = [
    "_role_from_msg",
    "_row_to_msg",
    "_msg_to_jsonl_dict",
    "_jsonl_dict_to_msg",
    "_load_jsonl",
    "_save_jsonl",
]
