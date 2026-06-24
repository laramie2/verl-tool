import re
from collections.abc import Iterable
from typing import Any

import numpy as np


TYPE_QUERY_RE = re.compile(
    r"^type\s+(?:\[|<)\d+(?:\]|>)\s+\[([^\]]+)\](?:\s+\[[01]\])?",
    re.IGNORECASE | re.DOTALL,
)
ACTION_BLOCK_RE = re.compile(r"```(.*?)```|<action>(.*?)</action>", re.DOTALL)


def normalize_ground_truths(value: Any) -> list[str]:
    """Convert dataset ground-truth containers into non-empty strings."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        result = []
        for item in value:
            result.extend(normalize_ground_truths(item))
        return result
    text = str(value).strip()
    return [text] if text else []


def normalize_match_text(text: Any) -> str:
    text = str(text or "").casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def ground_truth_match_score(text: Any, ground_truths: Any) -> float:
    """Return 1 when a normalized GT alias occurs in text, otherwise 0."""
    normalized_text = normalize_match_text(text)
    if not normalized_text:
        return 0.0
    padded_text = f" {normalized_text} "
    for answer in normalize_ground_truths(ground_truths):
        normalized_answer = normalize_match_text(answer)
        if normalized_answer and f" {normalized_answer} " in padded_text:
            return 1.0
    return 0.0


def extract_browser_action(action_text: Any) -> str:
    text = str(action_text or "").strip()
    matches = list(ACTION_BLOCK_RE.finditer(text))
    if not matches:
        return text
    match = matches[-1]
    return (match.group(1) or match.group(2) or "").strip()


def extract_search_query(action_text: Any) -> str:
    action = extract_browser_action(action_text)
    match = TYPE_QUERY_RE.match(action)
    return match.group(1).strip() if match else ""


def _query_tokens(query: str) -> set[str]:
    normalized = normalize_match_text(query)
    words = normalized.split()
    if len(words) > 1:
        return set(words)
    compact = normalized.replace(" ", "")
    return {compact[i : i + 2] for i in range(max(0, len(compact) - 1))} or ({compact} if compact else set())


def atomic_query_scores(
    tool_interact_info: Any,
    min_chars: int = 3,
    max_chars: int = 120,
    similarity_threshold: float = 0.8,
) -> dict[int, float]:
    """Score valid, concise and non-duplicate submitted search queries by turn."""
    if isinstance(tool_interact_info, np.ndarray):
        tool_interact_info = tool_interact_info.tolist()
    seen: list[set[str]] = []
    scores: dict[int, float] = {}
    for info in tool_interact_info or []:
        if not isinstance(info, dict):
            continue
        turn = info.get("action_turn_index")
        query = extract_search_query(info.get("action", ""))
        if not isinstance(turn, (int, np.integer)) or turn < 0 or not query:
            continue
        compact_len = len(normalize_match_text(query).replace(" ", ""))
        tokens = _query_tokens(query)
        valid_length = min_chars <= compact_len <= max_chars
        max_similarity = 0.0
        for previous in seen:
            union = tokens | previous
            similarity = len(tokens & previous) / len(union) if union else 1.0
            max_similarity = max(max_similarity, similarity)
        is_novel = max_similarity < similarity_threshold
        scores[int(turn)] = 1.0 if valid_length and is_novel else -1.0
        if valid_length and is_novel:
            seen.append(tokens)
    return scores


def refinement_candidate_text(generation: Any) -> str:
    """Use extracted conclusions and final answers, falling back to reasoning text."""
    text = str(generation or "")
    conclusions = re.findall(r"<conclusion>(.*?)</conclusion>", text, flags=re.IGNORECASE | re.DOTALL)
    stops = re.findall(r"```stop\s*\[([^\]]*)\]```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [part.strip() for part in conclusions + stops if part.strip()]
    if candidates:
        return "\n".join(candidates)
    return ACTION_BLOCK_RE.sub(" ", text)
