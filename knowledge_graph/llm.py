"""LLM interaction module for OpenAI-compatible API endpoints."""

from __future__ import annotations

import json
import re

import requests


def call_llm(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint.

    Args:
        prompt: The user prompt to send.
        base_url: The API base URL (e.g. http://localhost:11434/v1).
        model: The model name to use.
        api_key: Optional API key.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.

    Returns:
        The assistant's response text.
    """
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a knowledge extraction assistant. "
                    "You extract structured knowledge from text in the exact format requested. "
                    "Always respond with valid JSON only, no additional text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"]


def extract_triplets(text_chunk: str, base_url: str, model: str, api_key: str = "") -> list[dict]:
    """Extract Subject-Predicate-Object triplets from a text chunk.

    Returns a list of dicts with keys: subject, predicate, object.
    """
    prompt = f"""Analyze the following text and extract all knowledge as Subject-Predicate-Object (SPO) triplets.

Return a JSON array where each element has exactly three keys:
- "subject": the entity performing or being described
- "predicate": the relationship or action
- "object": the target entity or value

Extract as many meaningful triplets as possible. Be specific with entity names.

Text:
\"\"\"
{text_chunk}
\"\"\"

Return ONLY a valid JSON array, nothing else."""

    raw = call_llm(prompt, base_url, model, api_key)
    return _parse_json_array(raw)


def standardize_entities(
    triplets: list[dict], base_url: str, model: str, api_key: str = ""
) -> list[dict]:
    """Use LLM to standardize entity names across triplets."""
    if not triplets:
        return triplets

    entities = set()
    for t in triplets:
        entities.add(t.get("subject", ""))
        entities.add(t.get("object", ""))
    entities.discard("")

    if len(entities) < 2:
        return triplets

    entity_list = sorted(entities)

    prompt = f"""Below is a list of entity names extracted from a document. Many of these refer to the same real-world entity but with different names, abbreviations, or phrasings.

Create a mapping from each entity to its canonical (standardized) form. Group duplicates and variants together under the most descriptive canonical name.

Entities:
{json.dumps(entity_list)}

Return a JSON object where keys are the original entity names and values are the canonical names.
For entities that don't need standardization, map them to themselves.

Return ONLY a valid JSON object, nothing else."""

    raw = call_llm(prompt, base_url, model, api_key, temperature=0.0)
    mapping = _parse_json_object(raw)

    if not mapping:
        return triplets

    standardized = []
    for t in triplets:
        new_t = {
            "subject": mapping.get(t["subject"], t["subject"]),
            "predicate": t["predicate"],
            "object": mapping.get(t["object"], t["object"]),
        }
        standardized.append(new_t)

    return standardized


def infer_relationships(
    triplets: list[dict],
    communities: list[set[str]],
    base_url: str,
    model: str,
    api_key: str = "",
) -> list[dict]:
    """Use LLM to infer relationships between disconnected communities."""
    if len(communities) < 2:
        return []

    community_summaries = []
    for i, comm in enumerate(communities):
        sample = sorted(comm)[:10]
        community_summaries.append({"community": i, "entities": sample})

    prompt = f"""Below are entity communities from a knowledge graph. These communities are disconnected from each other.

Communities:
{json.dumps(community_summaries, indent=2)}

Analyze these communities and infer plausible relationships that could connect them.
Return a JSON array of SPO triplets with keys: "subject", "predicate", "object".
Only infer relationships that are logically sound based on the entity names.

Return ONLY a valid JSON array, nothing else."""

    raw = call_llm(prompt, base_url, model, api_key)
    inferred = _parse_json_array(raw)

    for t in inferred:
        t["inferred"] = True

    return inferred


def _parse_json_array(raw: str) -> list[dict]:
    """Parse a JSON array from LLM output, handling markdown code blocks."""
    raw = raw.strip()
    # Remove markdown code fences
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def _parse_json_object(raw: str) -> dict:
    """Parse a JSON object from LLM output, handling markdown code blocks."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}
