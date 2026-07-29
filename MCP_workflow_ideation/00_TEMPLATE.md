# MCP Workflow Ideation — Planning Worksheet

> Use this sheet to plan **one** agentic workflow that uses an MCP server, before
> you build it.

---

## 1. The idea, in one sentence
> _What will the agent let a user do?_

`__________________________________________________________________`

## 2. Goal & example prompts
- **Goal:** `__________________________________________________________`
- **Example user prompts** (write 2–3 real questions you'd type):
  1. `________________________________________________________________`
  2. `________________________________________________________________`
  3. `________________________________________________________________`

## 3. Why is this *agentic*?
> _An agent chooses **which** tool to call and **when**. What is the decision here?_
> _(If the agent always does the same single step, it's a chain, not an agent.)_

`__________________________________________________________________`

## 4. MCP server(s)

Useful links to find MCP Server: 
- https://github.com/mcp 
- https://smithery.ai 
- https://www.pulsemcp.com 
- https://mcpservers.org 
- https://mcpsafe.org 


| Field | Value |
|---|---|
| Name / provider | `__________________` |
| What it does | `__________________` |
| Transport | ☐ Remote (URL)  ☐ Local (stdio/command) |
| URL **or** command | `__________________` |
| Auth & cost | `__________________` (free? key? billing/credit-card?) |
| Tools it exposes | `__________________` (confirm live with **Refresh Tools**) |
| **Trust check** | ☐ First-party/vendor  ☐ Verified namespace  ☐ Safety-scored |

> Safe-selection rule of thumb: **prefer remote, free, first-party servers.**

## 5. Other Langflow components needed
> _Tick what you'll wire up._

☐ Chat Input ☐ Agent ☐ LLM (AISC hub) ☐ Prompt/System message
☐ Memory ☐ MCPTools node(s) ☐ Chat Output ☐ Other: `____________`

## 6. Flow sketch
> _Draw the boxes and arrows. A minimal MCP agent is almost always:_

```
Chat Input ─► Agent ─► Chat Output
               ▲ tools
               └── MCPTools node  
```

## 7. One-time setup steps
1. `________________________________________________________________`
2. `________________________________________________________________`
3. `________________________________________________________________`

## 8. Try it — success test
> _One concrete prompt + what a correct answer looks like._

`__________________________________________________________________`

## 9. Difficulty & likely gotchas
- **Difficulty:** ⭐ / ⭐⭐ / ⭐⭐⭐ / ⭐⭐⭐⭐
- **Gotchas / open questions:** `_____________________________________`
