"""Undercover Bench: auditable game controllers and experiment runner."""

from .games import SecretHitler, LiarsBar
from .ledger import Ledger
from .rewards import organizational_return, audit_score
from .trainable_backend import ModalTrainablePolicy

__all__ = [
    "SecretHitler",
    "LiarsBar",
    "Ledger",
    "organizational_return",
    "audit_score",
    "ModalTrainablePolicy",
]
