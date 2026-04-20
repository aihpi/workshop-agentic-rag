# Langflow Custom Components

## `SourceDocumentsBuilder`

Purpose:
- turn Qdrant retrieval rows into deterministic `source_documents`
- preserve file/page metadata for Chainlit citations
- build numbered context blocks for the answer prompt

### Wiring

1. Connect `QdrantVectorStoreComponent-QpAYx.dataframe` to `SourceDocumentsBuilder.input_data`.
2. Connect `SourceDocumentsBuilder.context_message` to your answer prompt context input.
3. Leave `SourceDocumentsBuilder.source_documents` as a terminal output so Langflow includes it in the `/api/v1/run/...` response.

### Prompt guidance

In the answer prompt, tell the model to cite using the numbered blocks:

```txt
Nutze fuer Belege nur die nummerierten Quellenbloecke aus dem Kontext.
Zitiere im Format: Quelle <n>: <Datei> (S.<page>-<page_end>)
Schreibe am Ende:

Anschlussfragen:
1. ...
2. ...
3. ...
```

### Expected API shape

This component emits:

```json
{
  "source_documents": [
    {
      "page_content": "...",
      "metadata": {
        "file": "standard_200_2.pdf",
        "page": 12,
        "page_end": 13,
        "section_title": "Basis-Absicherung"
      }
    }
  ]
}
```

That shape is already accepted by Chainlit's Langflow parser.

## `AnswerEnvelopeBuilder`

Purpose:
- expose both `answer_text` and `source_documents` in one terminal API payload
- make Chainlit citations work when Langflow would otherwise return only the final chat message

### Wiring

1. Connect your final answer message to `AnswerEnvelopeBuilder.answer_input`.
2. Connect `SourceDocumentsBuilder.source_documents` to `AnswerEnvelopeBuilder.source_documents_input`.
3. Leave `AnswerEnvelopeBuilder.envelope` as the terminal output for the API flow.
4. Do not rely on `Chat Output` alone if you want Chainlit citations. A plain chat message response does not contain structured `source_documents`.

### Expected API shape

This component emits:

```json
{
  "answer_text": "Die Antwort ...",
  "source_documents": [
    {
      "page_content": "...",
      "metadata": {
        "file": "IT_Grundschutz_Kompendium_Edition2023.pdf",
        "page": 25,
        "page_end": 26,
        "section_title": "Prozess-Bausteine"
      }
    }
  ]
}
```

Chainlit already parses both fields from the Langflow API response.

## `EnvelopeOutput`

Purpose:
- act as a dedicated terminal output node for Langflow API responses
- send `answer_text` and `source_documents` back together without converting them into plain chat text

### Wiring

1. Connect your final answer message to `EnvelopeOutput.answer_input`.
2. Connect `SourceDocumentsBuilder.source_documents` to `EnvelopeOutput.source_documents_input`.
3. Leave `EnvelopeOutput.envelope` as a terminal output.
4. If you still want a visible answer in the Langflow playground, keep `Chat Output` connected in parallel.

### Notes

- `Chat Output` stringifies its input into a `Message`. That is fine for the visible answer, but it does not preserve structured `source_documents` for Chainlit citations.
- `EnvelopeOutput` should be used for the API payload that Chainlit consumes.
- Chainlit already calls Langflow with `output_type="any"`, so terminal `EnvelopeOutput` payloads can be returned alongside chat output.

## `AdaptiveQdrantRetriever`

Purpose:
- retry retrieval with `top_k` values such as `3,5,8,10`
- accept the first attempt that looks good enough
- optionally return no results if every attempt looks weak
- optionally let a small LLM judge whether the retrieved snippets are actually enough
- optionally rewrite the search query and retry when a judged round stays insufficient

### Wiring

1. Replace the plain Qdrant retriever with `AdaptiveQdrantRetriever`.
2. Connect the same embedding node to `AdaptiveQdrantRetriever.embedding`.
3. Optionally connect a small `LanguageModel` to `AdaptiveQdrantRetriever.judge_llm`.
4. Optionally connect a small `LanguageModel` to `AdaptiveQdrantRetriever.rewrite_llm`.
5. Connect the routed user question to `AdaptiveQdrantRetriever.search_query`.
6. Connect `AdaptiveQdrantRetriever.search_results` or `.dataframe` to `SourceDocumentsBuilder.input_data`.
7. Optionally inspect `AdaptiveQdrantRetriever.attempt_debug` while tuning thresholds.

### Notes

- Default schedule: `3,5,8,10`
- Default behavior: if every attempt looks weak, return no hits
- You can enable `Return Best Effort Results` if you prefer the last attempt anyway
- If `Judge LLM` is connected, its `SUFFICIENT` / `INSUFFICIENT` verdict becomes the primary stop decision.
- `attempt_debug` now shows both heuristic metrics and judge metadata (`judge_decision`, `judge_reason`, `decision_source`).
- Use a cheap, low-temperature model for the judge. It only needs to classify retrieval quality, not answer the user.
- If `Rewrite On Insufficient` is enabled, the component can rewrite the query and run another full `top_k` schedule.
- `Rewrite LLM` is optional. If omitted, the component reuses `Judge LLM` for rewriting.
- `attempt_debug` now also shows `rounds`, `rewrites`, `selected_query`, and `stop_reason`.

## `TextOutput`

Purpose:
- expose plain text as a terminal Langflow API output
- return a `Message` object from a custom component when you want an additional text-only API payload

### Wiring

1. Connect the text you want to expose to `TextOutput.input_value`.
2. Leave `TextOutput.text` as a terminal output so Langflow includes it in the `/api/v1/run/...` response.

### Notes

- This is useful for extra text outputs or debugging.
- This does not replace `source_documents` for Chainlit citations.
