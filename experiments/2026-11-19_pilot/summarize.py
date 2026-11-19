"""Summarize pilot arms: accuracy (when gold labels exist), cost, latency, searches, failures."""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

# gpt-6.1-terra list prices per token (standard tier); flex is half.
PRICE_IN, PRICE_CACHED, PRICE_OUT = 1.00e-6, 0.05e-6, 5.00e-6
SEARCH_PRICE = {"brave": 5.0e-3, "exa": 7.0e-3, "openai": 10.0e-3, "knowledge_store": 0.0, None: 0.0}

here = Path(__file__).parent
gold = {row.get("claim_id", i): row["label"] for i, row in enumerate(json.loads((here / "../../data/averitec/dev_20.json").read_text()))}

arms = sys.argv[1:] or sorted(p.removesuffix(".runinfo.json") for p in glob.glob(str(here / "*.runinfo.json")))
print(f"{'arm':28} {'n':>2} {'acc':>5} {'LLM$/cl':>8} {'srch$/cl':>8} {'srch/cl':>7} {'s/claim':>7} {'p90 s':>6} {'fails':>5} {'tok/cl':>7} {'cache%':>6}")
for arm in arms:
    info = json.loads(Path(f"{arm}.runinfo.json").read_text())
    settings = info["settings"]
    flex = settings.get("service_tier") == "flex"
    provider = settings.get("search_provider")
    claims = info["claims"]
    costs, searches, durations, tokens, cache, failed, correct, labelled = [], [], [], [], [], 0, 0, 0
    for c in claims:
        u = c.get("usage") or {}
        inp, cached, out = u.get("input_tokens", 0), u.get("cached_input_tokens", 0), u.get("output_tokens", 0)
        cost = (inp - cached) * PRICE_IN + cached * PRICE_CACHED + out * PRICE_OUT
        costs.append(cost * (0.5 if flex else 1.0))
        tokens.append(u.get("total_tokens", 0))
        cache.append(u.get("cache_hit_rate") or 0)
        failed += (c.get("failed_checks") or 0) + (1 if c.get("error") else 0) + (u.get("failed_attempts") or 0)
        durations.append(c.get("duration_seconds") or 0)
        art = sorted(glob.glob(str(here / f"art_{Path(arm).name}" / f"*{c['claim_id']}*.json")))
        n_search = 0
        if art:
            summary = json.loads(Path(art[-1]).read_text()).get("retrieval_summary") or {}
            n_search = summary.get("searches", 0)
        searches.append(n_search)
        if "veritas" not in arm and c["claim_id"] in gold:
            labelled += 1
            correct += c.get("verdict") == gold[c["claim_id"]]
    p90 = sorted(durations)[max(0, int(0.9 * len(durations)) - 1)]
    acc = f"{correct}/{labelled}" if labelled else "-"
    print(
        f"{Path(arm).name:28} {len(claims):>2} {acc:>5} {st.mean(costs):>8.4f} "
        f"{st.mean(searches) * SEARCH_PRICE.get(provider, 0):>8.4f} {st.mean(searches):>7.1f} "
        f"{st.mean(durations):>7.0f} {p90:>6.0f} {failed:>5} {st.mean(tokens) / 1e3:>6.0f}k {100 * st.mean(cache):>5.0f}%"
    )
