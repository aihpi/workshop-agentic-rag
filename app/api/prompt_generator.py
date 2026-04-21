"""Generate per-KB system prompts and starter questions from sample chunks.

The reference prompt (usually `system.md`) pins the output shape — section
headers, citation format, Anschlussfragen block — so generated prompts stay
plug-compatible with the chat loop's citation rendering.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from core.llm import chat
from kb.rag_tool import _get_client


def _sample_chunks(collection: str, n: int = 30) -> list[dict[str, Any]]:
    client = _get_client()
    try:
        points, _ = client.scroll(
            collection_name=collection,
            limit=max(n * 3, 60),
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return []
    payloads = [p.payload for p in points if getattr(p, "payload", None)]
    if len(payloads) > n:
        payloads = random.sample(payloads, n)
    return payloads


def _format_samples(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for c in chunks:
        title = c.get("section_title") or c.get("file") or "Ausschnitt"
        page_start = c.get("page_start")
        page_end = c.get("page_end")
        pages = ""
        if page_start and page_end and page_start != page_end:
            pages = f" (S.{page_start}-{page_end})"
        elif page_start:
            pages = f" (S.{page_start})"
        text = (c.get("text") or "").strip()
        if len(text) > 700:
            text = text[:700] + "…"
        parts.append(f"### {title}{pages}\n{text}")
    return "\n\n".join(parts)


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


async def generate_system_prompt(collection: str, reference: str) -> str:
    chunks = _sample_chunks(collection, n=25)
    if not chunks:
        raise RuntimeError(
            "Keine Dokumente in dieser Wissensdatenbank gefunden. "
            "Bitte zuerst Dokumente hochladen."
        )
    context = _format_samples(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein Meta-Prompt-Architekt. Du erstellst System-Prompts "
                "für RAG-Chatbots, die Antworten ausschließlich aus "
                "bereitgestellten Dokumenten ableiten."
            ),
        },
        {
            "role": "user",
            "content": (
                "Erstelle einen neuen System-Prompt im Stil und Aufbau des "
                "Referenz-Prompts, thematisch zugeschnitten auf die "
                "Inhalte der Dokumentenausschnitte.\n\n"
                "MUSS-REGELN:\n"
                "- Sprache: Deutsch, Markdown-Überschriften wie in der Referenz.\n"
                "- Abschnitte 'IDENTITÄT UND ZIEL', 'SCHRITTE', 'AUSGABE', "
                "'ANSCHLUSSFRAGEN-FORMAT' müssen in dieser Reihenfolge vorhanden sein.\n"
                "- Das Quellenformat im Fließtext MUSS identisch zur Referenz "
                "übernommen werden: `Quelle <Nummer>: <Abschnittstitel> (S.<Start>-<Ende>)` "
                "bzw. `(S.<Start>)` bei Einzelseiten. Keine eckigen/geschweiften "
                "Klammern um das Token.\n"
                "- Der gesamte Abschnitt 'ANSCHLUSSFRAGEN-FORMAT' (Header, genau 3 "
                "Fragen, jeweils mit `?`, Formatvorgaben) muss wörtlich aus der "
                "Referenz übernommen werden.\n"
                "- Identität, Zielgruppe und Themenschwerpunkt an den Inhalt der "
                "Dokumente anpassen (nicht die BSI-Identität übernehmen, sofern die "
                "Dokumente ein anderes Thema behandeln).\n\n"
                "REFERENZ-PROMPT (Stil/Struktur):\n"
                "---\n"
                f"{reference}\n"
                "---\n\n"
                "DOKUMENTENAUSSCHNITTE:\n"
                "---\n"
                f"{context}\n"
                "---\n\n"
                "Gib ausschließlich den neuen System-Prompt als Markdown aus — "
                "ohne Einleitung, ohne Code-Fences, ohne Kommentar."
            ),
        },
    ]
    response = await chat(messages)
    content = (response.choices[0].message.content or "").strip()
    content = _FENCE_RE.sub("", content).strip()
    if not content:
        raise RuntimeError("Leere Antwort vom LLM erhalten.")
    return content


async def generate_starters(
    collection: str,
    reference: str,
    n_starters: int = 4,
) -> list[dict[str, str]]:
    chunks = _sample_chunks(collection, n=25)
    if not chunks:
        raise RuntimeError(
            "Keine Dokumente in dieser Wissensdatenbank gefunden. "
            "Bitte zuerst Dokumente hochladen."
        )
    context = _format_samples(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "Du erstellst Starterfragen für einen RAG-Chatbot, die den "
                "Nutzer zu sinnvollen Einstiegsfragen im Thema führen."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Erstelle genau {n_starters} Starterfragen für einen Chatbot, "
                "der Fragen auf Basis der unten aufgelisteten Dokumente "
                "beantwortet.\n\n"
                "Antworte als reines JSON-Array mit Objekten der Form "
                "{\"label\": \"...\", \"message\": \"...\"}.\n"
                "- label: Kurzbezeichnung (max. 4 Wörter, als Button-Text).\n"
                "- message: konkrete, beantwortbare Beispielfrage aus dem "
                "Themenbereich (max. 140 Zeichen).\n"
                "- Sprache: Deutsch.\n"
                "- Fragen sollen die Breite der Dokumente abdecken, nicht nur "
                "einen einzelnen Abschnitt.\n\n"
                "Gib ausschließlich JSON aus — keinen Fließtext, keine "
                "Code-Fences.\n\n"
                "REFERENZ-PROMPT (zum Kontext der Domäne):\n---\n"
                f"{reference[:1500]}\n---\n\n"
                "DOKUMENTENAUSSCHNITTE:\n"
                "---\n"
                f"{context}\n"
                "---"
            ),
        },
    ]
    response = await chat(messages)
    content = (response.choices[0].message.content or "").strip()
    content = _FENCE_RE.sub("", content).strip()
    try:
        items = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Konnte die Starter-Antwort nicht als JSON parsen: {exc}"
        ) from exc
    if not isinstance(items, list):
        raise RuntimeError("Starter-Antwort war kein JSON-Array.")
    result: list[dict[str, str]] = []
    for item in items[:n_starters]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()[:120]
        message = str(item.get("message") or "").strip()[:2000]
        if message:
            result.append({"label": label, "message": message})
    if not result:
        raise RuntimeError("Keine gültigen Starterfragen im LLM-Ergebnis gefunden.")
    return result
