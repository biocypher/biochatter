# 0001 — LLM backend and orchestration: how load-bearing should LangChain be?

Status: Draft for discussion
Date: 2026-06-24
Related: #334 (LangChain 1.x upgrade), #297 (structured `query()` return type)

## Question

biochatter must support hosted models (OpenAI, Anthropic, Gemini) and local models
(Ollama, vLLM, xinference) with little per-provider code and low maintenance. The stack
currently leans on LangChain/LangGraph, and #334 upgrades it to LangChain >=1.x. Should
LangChain stay the load-bearing spine, or should we route model calls through a thin spine
over the OpenAI-compatible API and keep LangChain only at the edges (retrievers,
integrations)?

## Two layers (often conflated)

- Serving backend: runs the model. vLLM, SGLang, llama.cpp (`llama-server`), Ollama, TGI,
  LMStudio. Ollama/LMStudio/Jan are wrappers over the llama.cpp engine.
- Orchestration: how the app talks to models. LangChain/LangGraph, or thinner options
  (LiteLLM, Pydantic AI, OpenAI Agents SDK, native SDK).

They are complementary, not competing.

## Findings

Sourced and adversarially verified (3 angles, 16 sources, 25 claims verified, 19 confirmed,
6 refuted). Confidence noted per item.

For a thin OpenAI-compatible spine:

1. vLLM, SGLang, llama.cpp, and Ollama all expose OpenAI `/v1` endpoints. Backend choice
   becomes a deployment detail, not a code dependency. (high)
2. LiteLLM gives one OpenAI-format interface to 100+ hosted providers and local backends
   (Ollama, vLLM, NVIDIA NIM) via the `openai/` prefix, normalizing to a single response
   type. biochatter already ships litellm. (high)
3. vLLM provides grammar-based structured outputs over the same OpenAI-compatible server
   (`guided_json`, `guided_grammar`, xgrammar/guidance). Enforcement is grammar-based token
   masking, so it does not depend on a model's native tool-calling ability. (high; strong,
   not an absolute "valid JSON without post-processing" guarantee)
4. vLLM tool calling matches the OpenAI surface: named functions plus `auto`/`required`
   (>=0.8.3)/`none`. (high)

Cost of keeping LangChain load-bearing:

5. LangChain v1 is breaking work: core namespace reduced; chains/retrievers/embeddings moved
   to `langchain-classic` (security fixes only until Dec 2026); `create_react_agent` →
   `create_agent` (`prompt` → `system_prompt`); custom agent hooks rewritten to middleware.
   Each major bump repeats this. (high)
6. Documented maintainability liability: debugging requires reading LangChain internals;
   fragmented docs slow onboarding. (medium, community sentiment)
7. Current practice for simple/linear agents is a thin wrapper over an OpenAI-compatible
   endpoint; full frameworks earn their cost for human-in-the-loop, multi-agent, or large
   shared tool schemas. (medium)

Typed outputs / agent loops with local models:

8. Pydantic AI is model-agnostic (20+ providers incl. Ollama and LiteLLM), MIT-licensed, low
   lock-in. It can sit on top of the LiteLLM spine. (medium)

Risks of the thin spine (handle at the edges):

9. OpenAI compatibility is imperfect for local models: Ollama's `/v1` collapses the post-tool
   response into one chunk (no token streaming); some vLLM-served models (gemma) lack
   system-message support and need remapping to `user`. (high)
10. Tool-calling reliability is model-dependent: Mistral 7B mishandles parallel calls,
    InternLM2 is unstable, small Llama 3 models emit malformed calls. Local agent loops
    should not rely on native tool calling alone; grammar-constrained output (item 3) is more
    robust. (high)
11. LiteLLM normalization has per-provider gaps (Ollama `tool_calls`/`finish_reason`, Gemini
    raw tool tokens, Responses-API streaming). Test per backend; expect some shims. (medium)

## Recommendation

1. Use the OpenAI-compatible Chat Completions API as the integration point, and route model
   calls through a thin spine. LiteLLM (already shipped) is the pragmatic choice.
2. For local schema enforcement, use grammar-based structured outputs (vLLM `guided_*`) where
   the backend supports it, rather than native tool calling.
3. Consider Pydantic AI for typed agent loops, on top of the spine.
4. Keep LangChain at the edges. Even if #334 lands as-is, wrap its usage behind a
   biochatter-owned interface so it can be demoted later without another rewrite.
5. Budget for edge work: per-backend tests for tool calling and structured output; a fallback
   for Ollama (e.g. disable streaming-with-tools).

## Relationship to #334

#334 incurs the migration cost in item 5. This does not necessarily block it. Decision: land
#334 with LangChain as the spine, or reframe its call sites to route through a thin spine now.
Minimum ask: #334's LangChain usage sits behind a biochatter-owned interface.

## #297

Same axis at smaller scale. Recommendation: do not add a bespoke `QueryResult` type; use the
spine's normalized response plus Pydantic models, integrated with #334's `LLMConnectionError`
and dict `token_usage`.

## Open questions

- How coupled is the current code to LangChain abstractions (message objects,
  `with_structured_output`, retrievers, prompt templates)? Cost of demoting vs finishing #334?
- Which local backends/models must we support in production?
- Do we need LangGraph agent features (multi-agent, HITL, persistence)?
- Fallback for Ollama structured output / streaming-with-tools?

## Caveats

LangChain/LangGraph v1 is recent (GA late 2025); `create_agent` location and the
`langchain-classic` window are moving; vLLM/LiteLLM version-gated facts should be re-checked
against pinned versions. Not relied on (refuted in verification): "raw-SDK migration cuts code
40-60% / debug time 70-90%"; "provider SDKs absorbed all of LangChain's value by 2025"; the
Pydantic AI "23 bugs / 8-10 DX" benchmark; "vLLM guarantees valid JSON without
post-processing". This document does not include a read of biochatter's own code or #334's
diff; it is guidance to weigh against the actual in-repo coupling.

## Sources

- vLLM tool calling: https://docs.vllm.ai/en/latest/features/tool_calling/
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling (issue #9632)
- LiteLLM: https://github.com/BerriAI/litellm , https://docs.litellm.ai/docs/providers/openai_compatible
- LangChain v1 migration: https://docs.langchain.com/oss/python/migrate/langchain-v1
- Pydantic AI models: https://pydantic.dev/docs/ai/models/overview/
