"""Generic fuzzy resolvers for capability-aware execution validation."""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional, Sequence, Tuple


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def fuzzy_match(
    query: str,
    candidates: Iterable[str],
    *,
    cutoff: float = 0.84,
) -> Tuple[Optional[str], float]:
    """Resolve query to candidate with confidence score."""
    q = _norm(query)
    pool = [str(c).strip() for c in (candidates or []) if str(c).strip()]
    if not q or not pool:
        return None, 0.0

    direct = {str(c).lower(): c for c in pool}
    if str(query).strip().lower() in direct:
        return direct[str(query).strip().lower()], 1.0

    token_map = {_norm(c): c for c in pool}
    if q in token_map:
        return token_map[q], 0.98

    best = difflib.get_close_matches(q, list(token_map.keys()), n=1, cutoff=cutoff)
    if not best:
        return None, 0.0
    hit = best[0]
    score = difflib.SequenceMatcher(a=q, b=hit).ratio()
    return token_map.get(hit), float(score)


def fuzzy_match_with_aliases(
    query: str,
    canonical_candidates: Sequence[str],
    aliases: Optional[dict[str, str]] = None,
    *,
    cutoff: float = 0.84,
) -> Tuple[Optional[str], float]:
    """Resolve query by searching aliases then canonical candidates."""
    alias_map = {str(k).strip(): str(v).strip() for k, v in (aliases or {}).items() if str(k).strip() and str(v).strip()}
    if alias_map:
        alias_hit, alias_score = fuzzy_match(query, alias_map.keys(), cutoff=cutoff)
        if alias_hit and alias_hit in alias_map:
            return alias_map[alias_hit], alias_score
    return fuzzy_match(query, canonical_candidates, cutoff=cutoff)
