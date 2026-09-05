# Brief: mcp-server

## Problem

A CLI is useful to a human at a prompt, but the highest-value consumer of this archive is an LLM assistant that can pull relevant passages mid-conversation. Without an MCP surface, using the archive from Claude means copying and pasting search results by hand — which nobody sustains.

This project is deliberately retrieval-only, and MCP is what makes that decision pay off: the server returns passages, the client does the reasoning. The value is in exposing good retrieval over a standard protocol, not in building another chat interface.

## Current State

- No server exists.
- `FastMCP` with `stateless_http=True` and `transport="streamable-http"` is the confirmed current path in the MCP Python SDK; streamable HTTP supersedes SSE for production use.
- The retrieval API this server wraps is owned by `vector-index` and does not yet exist.

## Desired Outcome

- Any MCP client can connect over streamable HTTP and search the archive.
- Tools are shaped for an LLM caller: clear names, well-described parameters, sensible defaults, and results that carry enough source attribution to cite.
- Identical parameters produce identical results in the MCP server and the CLI.
- The server starts via `uvx` and stays responsive without holding the whole index in memory.

## Approach

A thin `FastMCP` server over the same query-orchestration API the CLI uses. Stateless HTTP mode, since retrieval is inherently stateless and it avoids session-lifecycle complexity.

Tool design is the substance here, not the transport. An LLM caller needs a small number of well-described tools rather than the CLI's full flag surface — exposing fifteen tuning knobs to a model produces worse calls, not better ones. The server should offer a primary search tool with good defaults and a handful of genuinely useful parameters, plus tools to fetch a document's full text and to report index status. Advanced tuning stays in the CLI where a human is driving.

Results must carry source path, author, title, and score so the client can attribute what it uses.

## Scope

- **In**: `FastMCP` server over streamable HTTP in stateless mode; `uvx` entry point; a search tool exposing a deliberately reduced parameter set with strong defaults; a document-retrieval tool for fetching full text or an expanded context window around a hit; an index-status tool; result schemas with full source attribution; localhost binding with a configurable port; graceful handling of a missing, empty, or stale index; structured error responses; startup that neither blocks nor pre-loads the corpus.
- **Out**: Retrieval logic (`vector-index`). Embedding (`npu-embedding-runtime`). Ingest. Triggering a re-index from MCP — indexing is a deliberate, long-running operation that belongs to the CLI. Answer generation, prompts, or sampling. Authentication, TLS, or multi-user support — localhost only. Remote or internet exposure.

## Boundary Candidates

- Server lifecycle and transport configuration
- Tool definitions and their schemas
- Result serialization and source attribution
- Error and empty-state handling

## Out of Boundary

- The stdio transport. Not requested; it is cheap to add later behind the same tool definitions if a client needs it, and the design should not preclude that.
- Deciding retrieval defaults — inherited from `vector-index`.
- Any write path into the index.

## Upstream / Downstream

- **Upstream**: `vector-index` for the query-orchestration API. `npu-embedding-runtime` for query embedding. Shared configuration conventions established by `search-cli`.
- **Downstream**: Any MCP client — Claude Code, Claude Desktop, or third-party tooling over HTTP.

## Existing Spec Touchpoints

- **Extends**: None. Greenfield.
- **Adjacent**: `search-cli` is the sibling surface. Both consume the same retrieval API and must not diverge in ranking behaviour. Configuration resolution is established by the CLI spec and reused here rather than reinvented.

## Constraints

- Streamable HTTP transport (user-selected). Stateless mode preferred.
- Launchable via `uvx`, subject to the same conda-versus-`uvx` risk tracked in `npu-embedding-runtime`.
- Localhost binding by default. This server exposes the contents of a personal archive and must not be reachable off-machine without a deliberate choice.
- Query embedding runs on the NPU per the active provider setting, so the server must hold or reach a warm embedder without re-compiling the model per request.
- Tool descriptions are part of the contract — an LLM chooses tools by reading them, so vague descriptions are a functional defect, not a documentation one.
- No network egress.
