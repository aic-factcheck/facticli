"""The one place that talks to the Agents SDK: client setup, search tools, and the stage runner."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlparse

from agents import (
    Agent,
    ModelSettings,
    RunHooks,
    Runner,
    Tool,
    WebSearchTool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.models.interface import Model
from agents.run_error_handlers import RunErrorHandlerInput, RunErrorHandlers
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from facticli.brave_search import build_brave_web_search_tool
from facticli.core.retrieval import OPEN_PAGE, RetrievalJournal, RetrievalRecord, get_journal
from facticli.core.search_cache import SearchCache
from facticli.core.usage import StageUsageEvent, apply_usage, track_usage
from facticli.exa_search import build_exa_web_search_tool
from facticli.core.images import ClaimImage
from facticli.knowledge_store import build_knowledge_store_search_tool
from facticli.skills import load_skill_prompt

ApiMode = Literal["chat_completions", "responses"]

HOSTED_SEARCH_PROVIDER = "openai"
# Ask the Responses API for the complete list of URLs each hosted search
# consulted, so cited URLs can be checked against it afterwards.
HOSTED_SEARCH_INCLUDE = "web_search_call.action.sources"

WRAP_UP_NOTICE = (
    "HARNESS NOTICE: The turn limit for this task has been reached and the search tools are now disabled. "
    "Do not request further tool calls. Return your final structured output now, based only on the evidence "
    "already retrieved above. Where that evidence does not settle a question, say so: use the `insufficient` "
    "signal or lower your confidence, and record what is still missing. Do not fill gaps from memory."
)


# --- client ----------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """Resolved OpenAI-compatible endpoint: key, default model, base URL, and API mode."""
    api_key: str
    model: str
    base_url: str | None
    api_mode: ApiMode


def _clean(value: str | None) -> str | None:
    return (value or "").strip() or None


def infer_api_mode(base_url: str | None) -> ApiMode:
    """OpenAI's own endpoint gets the Responses API; every other base URL gets Chat Completions."""
    if not base_url:
        return "responses"
    host = urlparse(base_url).netloc.lower()
    if host == "api.openai.com" or host.endswith(".api.openai.com"):
        return "responses"
    return "chat_completions"


def resolve_endpoint(*, requested_model: str | None, base_url: str | None) -> Endpoint:
    """Resolve key (OPENAI_API_KEY), model (flag > OPENAI_API_MODEL), and base URL (flag > OPENAI_API_BASE_URL)."""
    api_key = _clean(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model = _clean(requested_model) or _clean(os.getenv("OPENAI_API_MODEL"))
    if not model:
        raise RuntimeError("OPENAI_API_MODEL is not set. Export it or pass --model.")
    resolved_base_url = _clean(base_url) or _clean(os.getenv("OPENAI_API_BASE_URL"))
    return Endpoint(api_key=api_key, model=model, base_url=resolved_base_url, api_mode=infer_api_mode(resolved_base_url))


def configure_client(endpoint: Endpoint, *, max_retries: int = 3) -> None:
    """Install the endpoint as the Agents SDK default client.

    Transient errors (429, 5xx, timeouts, dropped connections) are retried by
    the OpenAI client itself, with exponential backoff that honours
    ``Retry-After``; the SDK leaves those retries on when no SDK retry policy
    is set.
    """
    client = AsyncOpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url, max_retries=max_retries)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api(endpoint.api_mode)
    set_tracing_disabled(True)


def build_model_settings(
    *,
    parallel_tool_calls: bool | None,
    reasoning_effort: str | None = None,
    response_include: list[str] | None = None,
    service_tier: str | None = None,
) -> ModelSettings:
    """Model settings with only what every OpenAI-compatible provider accepts.

    Temperature and similar knobs stay unset because providers do not accept
    them uniformly. Reasoning effort and service tier are sent only when set.
    """
    return ModelSettings(
        parallel_tool_calls=parallel_tool_calls,
        reasoning=Reasoning(effort=reasoning_effort) if reasoning_effort else None,  # type: ignore[arg-type]
        response_include=response_include,  # type: ignore[arg-type]
        extra_args={"service_tier": service_tier} if service_tier else None,
    )


# --- search tools ----------------------------------------------------------


def build_search_tools(
    *,
    search_provider: str,
    search_context_size: str,
    max_chars_per_field: int,
    max_total_chars: int,
    search_results_per_query: int,
    cache: SearchCache | None = None,
) -> list[Tool]:
    """The research stage's search tool for the chosen provider."""
    if search_provider == HOSTED_SEARCH_PROVIDER:
        return [WebSearchTool(search_context_size=search_context_size)]  # type: ignore[arg-type]
    shared = dict(max_chars_per_field=max_chars_per_field, max_total_chars=max_total_chars, default_count=search_results_per_query)
    if search_provider == "brave":
        return [build_brave_web_search_tool(**shared, cache=cache)]
    if search_provider == "exa":
        return [build_exa_web_search_tool(**shared, cache=cache)]
    if search_provider == "knowledge_store":
        return [build_knowledge_store_search_tool(**shared)]
    raise ValueError(f"Unsupported search provider: {search_provider}")


def record_hosted_search_calls(response: Any, journal: RetrievalJournal) -> None:
    """Copy the hosted web-search calls of one model response into the retrieval journal.

    Hosted search runs on the provider's side, so the harness never sees page
    text, only (when requested with ``HOSTED_SEARCH_INCLUDE``) the URLs each
    call consulted. That is enough for URL-level provenance; quotes cannot be
    checked. A search that reports no source list leaves the journal incomplete.
    """
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        action_type = getattr(action, "type", None)
        if action_type == "search":
            sources = getattr(action, "sources", None)
            if sources is None:
                journal.complete = False
            queries = getattr(action, "queries", None) or [getattr(action, "query", None)]
            journal.add(
                RetrievalRecord(
                    provider=HOSTED_SEARCH_PROVIDER,
                    query=" | ".join(query for query in queries if query) or None,
                    urls=[source.url for source in sources or [] if getattr(source, "url", None)],
                )
            )
        elif action_type in ("open_page", "find_in_page"):
            url = getattr(action, "url", None)
            journal.add(RetrievalRecord(provider=HOSTED_SEARCH_PROVIDER, kind=OPEN_PAGE, urls=[url] if url else []))


# --- stage runner ----------------------------------------------------------


def stage_input(payload: dict[str, Any], images: Sequence[ClaimImage] = ()) -> str | list[Any]:
    """The user message of a stage: the JSON payload, then each image after a line naming its id.

    Without images the message is the bare JSON string, exactly as before
    images existed, so text-only runs send byte-identical requests.
    """
    message = json.dumps(payload, indent=2, ensure_ascii=False)
    if not images:
        return message
    content: list[dict[str, Any]] = [{"type": "input_text", "text": message}]
    for image in images:
        content.append({"type": "input_text", "text": f"[{image.image_id}]"})
        content.append({"type": "input_image", "image_url": image.data_url, "detail": "auto"})
    return [{"role": "user", "content": content}]


class _LedgerHooks(RunHooks[Any]):
    """Keep the usage ledger and the retrieval journal current as model responses arrive.

    The SDK hands usage and raw responses back only with a successful result.
    Copying them per response means a run that later exceeds its turn limit,
    fails validation, or is cancelled by a timeout still pays into the ledger
    and leaves its hosted-search receipts.
    """

    def __init__(self, event: StageUsageEvent | None) -> None:
        self._event = event

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        if self._event is not None:
            apply_usage(self._event, context.usage)
        journal = get_journal()
        if journal is not None:
            record_hosted_search_calls(response, journal)


class StageRunner:
    """One stage's agent, with usage tracking and a soft landing at the turn limit.

    A tool-using run that reaches its turn limit gets one more model call with
    tools disabled, so the evidence it already paid for becomes a finding
    written on less evidence instead of a failed check. The call is recorded as
    its own usage event; if it fails as well, the turn-limit error propagates.
    """

    def __init__(
        self,
        *,
        name: str,
        stage: str,
        output_type: type[Any],
        model: str | Model,
        max_turns: int,
        reasoning_effort: str | None = None,
        tools: Sequence[Tool] = (),
        service_tier: str | None = None,
    ) -> None:
        hosted_search = any(isinstance(tool, WebSearchTool) for tool in tools)
        settings = build_model_settings(
            parallel_tool_calls=bool(tools),
            reasoning_effort=reasoning_effort,
            response_include=[HOSTED_SEARCH_INCLUDE] if hosted_search else None,
            service_tier=service_tier,
        )
        self._agent: Agent[None] = Agent(
            name=name,
            instructions=load_skill_prompt(stage),
            tools=list(tools),
            output_type=output_type,
            model=model,
            model_settings=settings,
        )
        # Same instructions and tool schemas, so the prompt prefix stays
        # cache-identical; the model is only forbidden to call a tool.
        self._wrap_up_agent = (
            self._agent.clone(model_settings=replace(settings, tool_choice="none", parallel_tool_calls=None))
            if tools
            else None
        )
        self._output_type = output_type
        self._max_turns = max_turns
        self._stage = stage
        self.model_label = model if isinstance(model, str) else type(model).__name__

    async def run(self, payload: dict[str, Any], images: Sequence[ClaimImage] = ()) -> tuple[Any, bool]:
        """Run the stage; return the typed output and whether the wrap-up call produced it."""
        message = stage_input(payload, images)
        wrap_up_agent = self._wrap_up_agent
        if wrap_up_agent is None:
            return await self._tracked_run(self._agent, message, max_turns=self._max_turns), False

        journal = get_journal()
        if journal is not None:
            # Function tools journal every result and hosted search reports its
            # sources, so from here on a URL the journal lacks was not retrieved.
            journal.complete = True
        wrapped_up = False

        async def wrap_up(data: RunErrorHandlerInput[None]) -> Any | None:
            nonlocal wrapped_up
            try:
                output = await self._tracked_run(
                    wrap_up_agent,
                    [*data.run_data.history, {"role": "user", "content": WRAP_UP_NOTICE}],
                    max_turns=1,
                    wrap_up=True,
                )
            except Exception:
                return None  # the SDK raises the turn-limit error; the failed call stays in the ledger
            wrapped_up = True
            return output

        output = await self._tracked_run(
            self._agent, message, max_turns=self._max_turns, error_handlers={"max_turns": wrap_up}
        )
        return output, wrapped_up

    async def _tracked_run(
        self,
        agent: Agent[None],
        message: str | list[Any],
        *,
        max_turns: int,
        wrap_up: bool = False,
        error_handlers: RunErrorHandlers[None] | None = None,
    ) -> Any:
        with track_usage(stage=self._stage, model=self.model_label, wrap_up=wrap_up) as event:
            result = await Runner.run(
                agent, message, max_turns=max_turns, hooks=_LedgerHooks(event), error_handlers=error_handlers
            )
            return result.final_output_as(self._output_type, raise_if_incorrect_type=True)
