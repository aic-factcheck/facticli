"use strict";

const SAMPLES = {
  cs: "Premiér Petr Fiala včera prohlásil, že česká ekonomika loni vzrostla o 2,3 procenta a že nezaměstnanost klesla na 3,1 procenta. Podle něj je to nejlepší výsledek za posledních deset let. Myslím, že vláda odvádí skvělou práci.",
  sk: "Minister financií Slovenskej republiky uviedol, že inflácia na Slovensku v minulom roku klesla pod 5 percent a že priemerná mzda stúpla o 8 percent. Dúfame, že tento priaznivý trend bude pokračovať aj naďalej.",
  pl: "Według raportu Głównego Urzędu Statystycznego bezrobocie w Polsce spadło w zeszłym roku do 5 procent, a PKB wzrosło o 3,2 procent. Premier Donald Tusk ogłosił, że rząd zbudował 12 tysięcy nowych mieszkań. To naprawdę imponujące osiągnięcie.",
};

const LANG_NAMES = {
  cs: "Čeština",
  sk: "Slovenčina",
  pl: "Polski",
  en: "English",
  de: "Deutsch",
  uk: "Українська",
};

const $ = (id) => document.getElementById(id);

const els = {
  input: $("input-text"),
  maxClaims: $("max-claims"),
  model: $("model"),
  baseUrl: $("base-url"),
  extractBtn: $("extract-btn"),
  clearBtn: $("clear-btn"),
  copyBtn: $("copy-json"),
  error: $("error"),
  results: $("results"),
  claimList: $("claim-list"),
  claimCount: $("claim-count"),
  langBadge: $("lang-badge"),
  emptyState: $("empty-state"),
  coverage: $("coverage"),
  coverageList: $("coverage-list"),
  excluded: $("excluded"),
  excludedList: $("excluded-list"),
};

let lastResult = null;

/* ---------- helpers ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fillList(container, items) {
  container.replaceChildren();
  for (const item of items) container.appendChild(el("li", null, item));
}

function setLoading(isLoading) {
  els.extractBtn.disabled = isLoading;
  els.extractBtn.classList.toggle("is-loading", isLoading);
  els.extractBtn.querySelector(".btn__label").textContent = isLoading
    ? "Analyzing…"
    : "Extract claims";
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
  els.results.hidden = true;
}

function clearError() {
  els.error.hidden = true;
}

function langLabel(code) {
  if (!code) return "—";
  const name = LANG_NAMES[code.toLowerCase()];
  return name ? `${name} · ${code}` : code;
}

/* ---------- rendering ---------- */

function renderClaims(claims) {
  els.claimList.replaceChildren();
  claims.forEach((claim, i) => {
    const li = el("li", "claim");

    li.appendChild(el("span", "claim__index", String(i + 1)));
    li.appendChild(el("p", "claim__text", claim.claim_text || ""));

    if (claim.source_fragment) {
      li.appendChild(el("blockquote", "claim__source", claim.source_fragment));
    }
    if (claim.checkworthy_reason) {
      const reason = el("p", "claim__reason");
      reason.appendChild(el("b", null, "Why"));
      reason.appendChild(el("span", null, claim.checkworthy_reason));
      li.appendChild(reason);
    }
    els.claimList.appendChild(li);
  });
}

function renderResult(result) {
  lastResult = result;
  clearError();

  const claims = Array.isArray(result.claims) ? result.claims : [];
  els.claimCount.textContent =
    claims.length === 1 ? "1 claim" : `${claims.length} claims`;
  els.langBadge.textContent = langLabel(result.detected_language);

  renderClaims(claims);
  els.emptyState.hidden = claims.length !== 0;

  const coverage = result.coverage_notes || [];
  els.coverage.hidden = coverage.length === 0;
  if (coverage.length) fillList(els.coverageList, coverage);

  const excluded = result.excluded_nonfactual || [];
  els.excluded.hidden = excluded.length === 0;
  if (excluded.length) fillList(els.excludedList, excluded);

  els.results.hidden = false;
  els.results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- actions ---------- */

async function extract() {
  const text = els.input.value.trim();
  if (!text) {
    showError("Please enter some text to analyze.");
    els.input.focus();
    return;
  }

  const body = {
    text,
    max_claims: Math.min(50, Math.max(1, parseInt(els.maxClaims.value, 10) || 12)),
  };
  const model = els.model.value.trim();
  const baseUrl = els.baseUrl.value.trim();
  if (model) body.model = model;
  if (baseUrl) body.base_url = baseUrl;

  setLoading(true);
  clearError();
  try {
    const resp = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `Request failed (HTTP ${resp.status}).`);
    }
    renderResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    setLoading(false);
  }
}

async function copyJson() {
  if (!lastResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
    const original = els.copyBtn.textContent;
    els.copyBtn.textContent = "Copied ✓";
    setTimeout(() => (els.copyBtn.textContent = original), 1400);
  } catch {
    showError("Could not copy to clipboard.");
  }
}

/* ---------- wiring ---------- */

els.extractBtn.addEventListener("click", extract);
els.copyBtn.addEventListener("click", copyJson);
els.clearBtn.addEventListener("click", () => {
  els.input.value = "";
  els.results.hidden = true;
  clearError();
  els.input.focus();
});

document.querySelectorAll(".chip[data-sample]").forEach((chip) => {
  chip.addEventListener("click", () => {
    els.input.value = SAMPLES[chip.dataset.sample] || "";
    els.input.focus();
  });
});

els.input.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    extract();
  }
});
