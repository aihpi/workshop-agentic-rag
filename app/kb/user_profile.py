"""User profile extraction and personalization utilities.

This module handles:
- Extracting user interest topics from chat history via LLM
- Computing topic embeddings for retrieval filtering
- Dynamically determining personalization balance per query
- Filtering retrieval results based on user profile
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chat.chat_history import (
    get_user_message_count,
    get_user_message_history,
    get_user_profile,
    upsert_user_profile,
)
from core.llm import chat, embed
from core.settings import (
    CHAT_DB_PATH,
    PERSONALIZATION_ENABLED,
    PROFILE_MIN_MESSAGES,
    PROFILE_RELEVANCE_THRESHOLD,
    PROFILE_TOPIC_LIMIT,
)


@dataclass
class UserProfile:
    """Represents extracted user profile for personalization."""

    user_id: str
    topics: list[str] = field(default_factory=list)
    topic_embeddings: list[list[float]] = field(default_factory=list)
    excluded_bausteine: list[str] = field(default_factory=list)
    message_count: int = 0

    def has_sufficient_history(self) -> bool:
        """Check if user has enough history for personalization."""
        return self.message_count >= PROFILE_MIN_MESSAGES

    def to_context_string(self) -> str:
        """Format topics as a string for prompt injection."""
        if not self.topics:
            return ""
        return ", ".join(self.topics)


TOPIC_EXTRACTION_PROMPT = """Analysiere die folgenden Benutzeranfragen an einen IT-Grundschutz-Chatbot und extrahiere die Hauptthemen, für die sich der Benutzer interessiert.

Benutzeranfragen:
{messages}

Aufgabe:
1. Identifiziere die wichtigsten Themen/Bereiche aus den Anfragen (z.B. "Webserver-Sicherheit", "Netzwerksegmentierung", "Cloud-Computing", "Authentifizierung")
2. Gib maximal {topic_limit} Themen zurück
3. Verwende prägnante, spezifische Begriffe aus dem IT-Grundschutz-Kontext

Antworte NUR mit einem JSON-Array der Themen, ohne weitere Erklärungen.
Beispiel: ["Webserver-Sicherheit", "Authentifizierung", "Netzwerksegmentierung"]

Themen:"""


BALANCE_DETERMINATION_PROMPT = """Du bist ein System, das entscheidet, wie stark die Personalisierung bei einer RAG-Anfrage gewichtet werden soll.

Benutzerrolle: {user_role}
Benutzer-Interessen: {user_topics}
Aktuelle Anfrage: {query}

Entscheide, wie stark die Personalisierung gewichtet werden soll (Fließkommawert zwischen 0.0 und 1.0):

- 1.0: Keine Personalisierung - Standard-Retrieval
  (für allgemeine Fragen oder wenn die Anfrage NICHT zu den Interessen/der Rolle passt)
  
- 0.7-0.9: Geringe Personalisierung
  (für Fragen mit schwachem Bezug zu den Interessen)
  
- 0.4-0.6: Ausgewogene Mischung
  (für Fragen, die teilweise mit den Interessen oder der Rolle zusammenhängen)
  
- 0.1-0.3: Starke Personalisierung
  (für Fragen mit deutlichem Bezug zu den Benutzer-Interessen)
  
- 0.0: Maximale Personalisierung
  (für Fragen, die exakt die Interessen und Rolle des Benutzers betreffen)

Berücksichtige auch die Rolle des Benutzers: Eine technische Anfrage sollte für "Durchführungsverantwortliche IT-Betrieb" stärker personalisiert werden als für "Institutsleitung".

Antworte NUR mit einer Dezimalzahl zwischen 0.0 und 1.0, ohne weitere Erklärungen."""


async def extract_user_topics(
    user_id: str,
    db_path: Path | None = None,
    force: bool = False,
) -> list[str]:
    """Extract interest topics from user's chat history using LLM.

    Args:
        user_id: The user identifier
        db_path: Path to chat database (defaults to CHAT_DB_PATH)
        force: Force re-extraction even if profile exists

    Returns:
        List of extracted topic strings
    """
    db = db_path or CHAT_DB_PATH

    # Check if we already have a recent profile
    if not force:
        existing = get_user_profile(db, user_id)
        if existing and existing.get("topics"):
            return existing["topics"]

    # Get user's recent messages
    messages = get_user_message_history(db, user_id, limit=100, role_filter="user")
    if not messages:
        return []

    # Format messages for prompt
    message_texts = "\n".join(
        f"- {msg['content'][:500]}" for msg in messages[:50]  # Limit context size
    )

    prompt = TOPIC_EXTRACTION_PROMPT.format(
        messages=message_texts,
        topic_limit=PROFILE_TOPIC_LIMIT,
    )

    try:
        response = await chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
        )
        content = response.choices[0].message.content or ""  # type: ignore[union-attr]
        content = content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        topics = json.loads(content)
        if isinstance(topics, list):
            return [str(t).strip() for t in topics if t][:PROFILE_TOPIC_LIMIT]
    except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
        print(f"[WARN] topic_extraction_failed: {e}")

    return []


async def update_user_profile(
    user_id: str,
    db_path: Path | None = None,
    force: bool = False,
) -> UserProfile:
    """Extract topics and update user profile in database.

    Args:
        user_id: The user identifier
        db_path: Path to chat database
        force: Force re-extraction

    Returns:
        Updated UserProfile instance
    """
    db = db_path or CHAT_DB_PATH

    # Get current message count
    message_count = get_user_message_count(db, user_id)

    # Check existing profile
    existing = get_user_profile(db, user_id)
    if existing and not force:
        # Only update if message count increased significantly (10+ new messages)
        existing_count = existing.get("message_count", 0)
        threshold = existing_count + 10
        if message_count < threshold:
            return UserProfile(
                user_id=user_id,
                topics=existing.get("topics", []),
                topic_embeddings=existing.get("topic_embeddings", []),
                excluded_bausteine=existing.get("excluded_bausteine", []),
                message_count=existing_count,
            )

    # Extract topics via LLM
    topics = await extract_user_topics(user_id, db, force=True)

    # Embed topics for similarity matching
    topic_embeddings: list[list[float]] = []
    if topics:
        try:
            topic_embeddings = await embed(topics)
        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            print(f"[WARN] topic_embedding_failed: {e}")

    # Store in database
    upsert_user_profile(
        db,
        user_id,
        topics=topics,
        topic_embeddings=topic_embeddings,
        message_count=message_count,
    )

    return UserProfile(
        user_id=user_id,
        topics=topics,
        topic_embeddings=topic_embeddings,
        excluded_bausteine=existing.get("excluded_bausteine", []) if existing else [],
        message_count=message_count,
    )


async def load_user_profile(
    user_id: str,
    db_path: Path | None = None,
) -> UserProfile | None:
    """Load user profile from database.

    Args:
        user_id: The user identifier
        db_path: Path to chat database

    Returns:
        UserProfile instance or None if not found
    """
    if not PERSONALIZATION_ENABLED:
        return None

    db = db_path or CHAT_DB_PATH
    data = get_user_profile(db, user_id)

    if not data:
        return None

    return UserProfile(
        user_id=user_id,
        topics=data.get("topics", []),
        topic_embeddings=data.get("topic_embeddings", []),
        excluded_bausteine=data.get("excluded_bausteine", []),
        message_count=data.get("message_count", 0),
    )


async def determine_balance(
    query: str,
    user_profile: UserProfile | None,
    user_role: str = "",
) -> float:
    """Dynamically determine personalization balance for a query.

    Uses LLM to decide how strongly to weight personalization based on
    whether the query relates to user's known interests and role.

    Args:
        query: The user's current query
        user_profile: User's profile with extracted topics
        user_role: The user's selected chat profile/role

    Returns:
        Balance value between 0.0 (full personalization) and 1.0 (no personalization)
    """
    # No personalization without profile or topics
    if not user_profile or not user_profile.topics:
        return 1.0

    # Not enough history for personalization
    if not user_profile.has_sufficient_history():
        return 1.0

    prompt = BALANCE_DETERMINATION_PROMPT.format(
        user_role=user_role or "Nicht angegeben",
        user_topics=", ".join(user_profile.topics),
        query=query,
    )

    try:
        response = await chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
        )
        content = response.choices[0].message.content or ""  # type: ignore[union-attr]
        content = content.strip()

        # Parse balance value
        balance = float(content)
        return max(0.0, min(1.0, balance))  # Clamp to [0, 1]
    except (ValueError, IndexError, KeyError, AttributeError) as e:
        print(f"[WARN] balance_determination_failed: {e}")
        return 1.0  # Default to no personalization on error


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def compute_profile_relevance(
    chunk_embedding: list[float],
    user_profile: UserProfile,
) -> float:
    """Compute relevance score of a chunk to user's interest profile.

    Args:
        chunk_embedding: Embedding vector of the retrieved chunk
        user_profile: User's profile with topic embeddings

    Returns:
        Relevance score between 0.0 and 1.0
    """
    if not user_profile.topic_embeddings:
        return 0.5  # Neutral score if no profile embeddings

    # Compute max similarity to any topic
    max_sim = 0.0
    for topic_emb in user_profile.topic_embeddings:
        sim = compute_similarity(chunk_embedding, topic_emb)
        max_sim = max(max_sim, sim)

    return max_sim


def filter_by_profile_relevance(
    results: list[dict[str, Any]],
    user_profile: UserProfile,
    threshold: float = PROFILE_RELEVANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Filter retrieval results by relevance to user profile.

    Removes chunks that are semantically distant from user's interests.

    Args:
        results: List of retrieval results with embeddings
        user_profile: User's profile
        threshold: Minimum relevance score to keep (default from settings)

    Returns:
        Filtered list of results
    """
    if not user_profile.topic_embeddings:
        return results

    filtered = []
    for result in results:
        embedding = result.get("embedding")
        if not embedding:
            # Keep results without embeddings
            filtered.append(result)
            continue

        relevance = compute_profile_relevance(embedding, user_profile)
        if relevance >= threshold:
            result["profile_relevance"] = relevance
            filtered.append(result)

    return filtered


def blend_retrieval_scores(
    base_results: list[dict[str, Any]],
    user_profile: UserProfile | None,
    balance: float = 0.5,
) -> list[dict[str, Any]]:
    """Blend base retrieval scores with profile relevance scores.

    Args:
        base_results: Results from standard retrieval with 'score' field
        user_profile: User's profile for computing relevance
        balance: Weight for base score (1.0 = only base, 0.0 = only personalized)

    Returns:
        Results with blended scores, re-sorted
    """
    if not user_profile or not user_profile.topic_embeddings or balance >= 1.0:
        return base_results

    for result in base_results:
        base_score = result.get("score", 0.0)
        embedding = result.get("embedding")

        if embedding:
            profile_relevance = compute_profile_relevance(embedding, user_profile)
        else:
            profile_relevance = 0.5  # Neutral

        # Blend scores
        blended_score = balance * base_score + (1 - balance) * profile_relevance
        result["original_score"] = base_score
        result["profile_relevance"] = profile_relevance
        result["score"] = blended_score

    # Re-sort by blended score
    return sorted(base_results, key=lambda x: x.get("score", 0), reverse=True)
