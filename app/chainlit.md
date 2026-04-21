# Agentic RAG — Workshop

Willkommen im Chat. Diese App zeigt, wie ein Agent mit Tool‑Calling, Vektorsuche (Qdrant) und LiteLLM ein RAG‑System aufbaut, das Quellen mit Zitationen ausgibt.

## Was du tun kannst

- **Fragen stellen** — der Agent entscheidet selbst, wann er die Wissensbasis (KB) abfragt, und antwortet mit Zitationen im Format `[Quelle N]` sowie Seitenangaben.
- **Quelle öffnen** — Klick auf eine Zitation öffnet das PDF in der rechten Seitenleiste auf der zitierten Seite.
- **Anschlussfragen** — am Ende einer Antwort werden passende Folgefragen vorgeschlagen.

## Eigene Wissensbasis

Über das Zahnrad oben rechts → **Einstellungen** kannst du:

- neue **Knowledge Bases (KBs)** anlegen und PDFs hochladen,
- den **System‑Prompt** und die **Starter‑Fragen** pro Nutzer anpassen,
- einzelne Dokumente oder ganze KBs wieder löschen.

Hochgeladene PDFs werden pro Dokument in Qdrant indexiert und stehen sofort dem Agenten zur Verfügung.

## Nützliche Slash‑Befehle

| Befehl | Zweck |
|---|---|
| `/history` | Letzte gespeicherte Chat‑Sitzungen auflisten |
| `/history <id>` | Nachrichten einer Sitzung anzeigen |
| `/export` | Aktuellen Chat als OpenAI‑JSON exportieren |
| `/export <id>` | Eine Sitzung exportieren |
| `/export all` | Alle Chats als JSONL exportieren |

## Geteilte Wissensbasis

Die App enthält eine gemeinsame, schreibgeschützte KB mit dem **IT‑Grundschutz‑Kompendium** und den **BSI‑Standards (200‑1 / 2 / 3 / 4)**. Der Agent wählt automatisch die passende KB aus dem Katalog.

---

Mehr Infos zur Architektur: siehe das Projekt‑README im Repo.
