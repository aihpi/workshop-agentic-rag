# Chat-Profile Konfiguration

Diese Dokumentation beschreibt, wie die Chat-Profile für den IT-Grundschutz Chatbot angepasst werden können.

## Übersicht

Chat-Profile ermöglichen es Benutzern, ihre Rolle über die **Chat-Einstellungen** auszuwählen. Die Einstellungen sind über das Zahnrad-Symbol in der Kopfzeile erreichbar. Basierend auf der gewählten Rolle werden:

1. **Antworten angepasst**: Der Chatbot erhält Kontext zur Benutzerrolle und passt Sprache und Fokus entsprechend an
2. **Personalisierung optimiert**: Die dynamische Balance zwischen Standard- und personalisiertem Retrieval berücksichtigt die Rolle

### Persistenz

Die gewählte Rolle wird **dauerhaft gespeichert** und bleibt auch nach Logout/Login erhalten. Dies ermöglicht eine konsistente Nutzungserfahrung über mehrere Sitzungen hinweg.

## Konfigurationsdatei

Die Profile werden in der Datei `apps/chainlit/chat_profiles.json` definiert.

### Grundstruktur

```json
{
  "profiles": [
    {
      "id": "eindeutige_id",
      "name": "Anzeigename im UI",
      "icon": "/public/icons/beispiel.svg",
      "description": "Kurze Beschreibung (eine Zeile)",
      "markdown_description": "Ausführliche Beschreibung mit **Markdown**",
      "relevant_bausteine": ["ISMS", "ORP", ...],
      "relevant_topics": ["Thema1", "Thema2", ...],
      "prompt_context": "Anweisung für den Chatbot..."
    }
  ],
  "default_profile": "id_des_standardprofils",
  "settings": { ... }
}
```

## Felder im Detail

### Pflichtfelder

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `id` | String | Eindeutige Kennung (Kleinbuchstaben, keine Leerzeichen) |
| `name` | String | Anzeigename im Profil-Auswahlmenü |
| `prompt_context` | String | Anweisung an den Chatbot zur Anpassung der Antworten |

### Optionale Felder

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `icon` | String | Pfad zum Icon (SVG empfohlen, relativ zu `/public/`) |
| `description` | String | Kurze Beschreibung (Fallback für `markdown_description`) |
| `markdown_description` | String | Ausführliche Beschreibung mit Markdown-Formatierung |
| `relevant_bausteine` | Array | IT-Grundschutz Bausteine relevant für diese Rolle |
| `relevant_topics` | Array | Typische Themen für diese Rolle |

## Beispiel: Neues Profil hinzufügen

Um ein neues Profil hinzuzufügen:

1. Öffnen Sie `apps/chainlit/chat_profiles.json`
2. Fügen Sie ein neues Objekt im `profiles`-Array hinzu:

```json
{
  "id": "datenschutz",
  "name": "Datenschutzbeauftragte",
  "icon": "/public/icons/lock.svg",
  "description": "Verantwortlich für den Datenschutz der Institution",
  "markdown_description": "Als **Datenschutzbeauftragte/r** erhalten Sie Antworten mit Fokus auf:\n- DSGVO-Konformität\n- Datenschutz-Folgenabschätzung\n- Technisch-organisatorische Maßnahmen",
  "relevant_bausteine": ["CON", "ORP", "APP"],
  "relevant_topics": [
    "DSGVO",
    "Datenschutz",
    "Personenbezogene Daten",
    "Löschkonzepte"
  ],
  "prompt_context": "Du antwortest einer Person in der Rolle Datenschutzbeauftragte/r. Fokussiere auf datenschutzrechtliche Aspekte, DSGVO-Konformität und technisch-organisatorische Maßnahmen. Weise auf Schnittstellen zwischen Informationssicherheit und Datenschutz hin."
}
```

3. Speichern Sie die Datei
4. Starten Sie die Anwendung neu

## Hinweise zum `prompt_context`

Der `prompt_context` wird dem System-Prompt hinzugefügt und beeinflusst direkt, wie der Chatbot antwortet.

### Bewährte Praktiken

- **Beginnen Sie mit der Rollenbezeichnung**: `"Du antwortest einer Person in der Rolle..."`
- **Definieren Sie den Fokus**: Was soll betont werden?
- **Geben Sie konkrete Anweisungen**: "Halte Antworten kompakt", "Vermeide zu technische Details"
- **Beschränken Sie die Länge**: 1-3 Sätze sind optimal

### Beispiele für verschiedene Rollen

**Technische Rolle:**
```
"Du antwortest einer Person in der Rolle Durchführungsverantwortliche/r IT-Betrieb. Gib konkrete, technische und praxisnahe Anleitungen zur Umsetzung von Sicherheitsmaßnahmen. Fokussiere auf Konfiguration, Härtung und operative Tätigkeiten."
```

**Management-Rolle:**
```
"Du antwortest einer Person in der Rolle Institutsleitung/Geschäftsführung. Fokussiere auf Governance, strategische Entscheidungen, Gesamtverantwortung und Ressourcenbereitstellung. Halte Antworten kompakt und entscheidungsorientiert. Vermeide zu technische Details."
```

## Icons hinzufügen

Icons müssen im Ordner `apps/chainlit/public/icons/` abgelegt werden.

### Empfehlungen
- Format: SVG (skalierbar, kleine Dateigröße)
- Größe: 24x24 px Viewbox
- Farbe: Monochromatisch (wird vom UI eingefärbt)

### Verfügbare Standard-Icons
- `/public/icons/shield.svg` - Schild
- `/public/icons/server.svg` - Server
- `/public/icons/briefcase.svg` - Aktentasche
- `/public/icons/settings.svg` - Zahnrad
- `/public/icons/building.svg` - Gebäude
- `/public/icons/search.svg` - Lupe
- `/public/icons/book.svg` - Buch

## Einstellungen (`settings`)

```json
"settings": {
  "allow_profile_change": true,
  "show_profile_in_response": false,
  "combine_with_chat_history": true
}
```

| Einstellung | Beschreibung |
|-------------|--------------|
| `allow_profile_change` | Ob Benutzer das Profil während einer Sitzung wechseln können |
| `show_profile_in_response` | Ob das aktive Profil in Antworten angezeigt wird |
| `combine_with_chat_history` | Ob Personalisierung aus Chat-Historie zusätzlich verwendet wird |

## Fehlerbehebung

### Profile werden nicht in den Einstellungen angezeigt
1. Prüfen Sie die JSON-Syntax (z.B. mit einem Online-Validator)
2. Stellen Sie sicher, dass das `profiles`-Array nicht leer ist
3. Prüfen Sie die Konsole auf Fehler beim Laden
4. Stellen Sie sicher, dass `chat_settings_location = "sidebar"` in `.chainlit/config.toml` gesetzt ist

### Icon wird nicht angezeigt
1. Prüfen Sie den Pfad (muss mit `/public/` beginnen)
2. Stellen Sie sicher, dass die Datei existiert
3. Prüfen Sie die Dateiberechtigungen

### Profil hat keine Auswirkung
1. Prüfen Sie, ob `prompt_context` gesetzt ist
2. Wechseln Sie das Profil in den Einstellungen (Zahnrad-Symbol)
3. Eine Bestätigungsmeldung erscheint nach dem Wechsel

## Technische Details

### Wie Profile funktionieren

1. **Bei Sitzungsstart**: Das zuletzt gewählte Profil wird aus der Datenbank geladen
2. **Chat-Einstellungen**: Benutzer können das Profil jederzeit über das Zahnrad-Symbol ändern
3. **Persistenz**: Änderungen werden in der SQLite-Datenbank (`user_profiles.selected_chat_profile`) gespeichert
4. **System-Prompt**: Der `prompt_context` wird als `## ROLLENKONTEXT` eingefügt
5. **Bei Retrieval**: Die Rolle wird an die Balance-Bestimmung übergeben

### Datenbank-Schema

Die Profilauswahl wird in der `user_profiles`-Tabelle gespeichert:
```sql
selected_chat_profile TEXT  -- Name des gewählten Profils
```

### Zusammenspiel mit Personalisierung

- **Chat-Profil**: Statische Rolleninformation (vom Benutzer gewählt, persistent)
- **User-Profile**: Dynamische Interessensextraktion (aus Chat-Historie)

Beide Mechanismen ergänzen sich:
- Das Chat-Profil gibt die grundlegende Perspektive vor
- Das User-Profile verfeinert basierend auf tatsächlichem Nutzungsverhalten

## Änderungshistorie

| Datum | Änderung |
|-------|----------|
| 2025-01 | Initiale Erstellung mit 5 Standardprofilen |
| 2026-03 | Profil-Auswahl in Chat-Einstellungen verschoben, Persistenz hinzugefügt |
