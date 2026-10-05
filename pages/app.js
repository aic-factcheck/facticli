/*
 * Thin client for the CEDMO claim extractor.
 * Extraction runs on the facticli backend (POST <API_BASE>/api/extract); this
 * page only collects input, forwards the shared access key as a bearer token
 * and renders the response. No model credential is shipped to the browser and
 * no model API is called from here.
 *
 * The backend URL comes from ./config.js (window.FACTICLI_CONFIG.apiBase),
 * written at deploy time, and can be overridden in the Advanced panel for
 * local testing against `python -m facticli.web`.
 */

const API_BASE = (
  (typeof window !== "undefined" && window.FACTICLI_CONFIG && window.FACTICLI_CONFIG.apiBase) || ""
).replace(/\/+$/, "");

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

const PW_STORAGE = "facticli_demo_pw";

const $ = (id) => document.getElementById(id);

const els = {
  input: $("input-text"),
  accessPw: $("access-pw"),
  maxClaims: $("max-claims"),
  apiBase: $("api-base"),
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

/* ---------- extraction ---------- */

function apiBase() {
  const override = els.apiBase.value.trim().replace(/\/+$/, "");
  return override || API_BASE;
}

async function callExtraction(text, maxClaims) {
  const base = apiBase();
  if (!base) {
    throw new Error(
      "No backend configured for this deployment. Set the API base URL in the Advanced panel.",
    );
  }
  const key = els.accessPw.value.trim();
  try {
    localStorage.setItem(PW_STORAGE, key);
  } catch {
    /* storage unavailable — key just won't persist */
  }

  let resp;
  try {
    resp = await fetch(`${base}/api/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify({ text, max_claims: maxClaims }),
    });
  } catch {
    throw new Error(`Could not reach the extraction service at ${base}.`);
  }

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    if (resp.status === 401) throw new Error("Incorrect access key.");
    if (resp.status === 429) throw new Error("Rate limit exceeded. Please wait and try again.");
    if (resp.status === 503) throw new Error(data.detail || "The service is not configured.");
    throw new Error(data.detail || `Request failed (HTTP ${resp.status}).`);
  }
  return data;
}

async function extract() {
  const text = els.input.value.trim();
  if (!text) {
    showError("Please enter some text to analyze.");
    els.input.focus();
    return;
  }
  if (!els.accessPw.value) {
    showError("Please enter the access key (ask the CEDMO team for it).");
    els.accessPw.focus();
    return;
  }

  const maxClaims = Math.min(50, Math.max(1, parseInt(els.maxClaims.value, 10) || 12));

  setLoading(true);
  clearError();
  try {
    renderResult(await callExtraction(text, maxClaims));
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

try {
  const saved = localStorage.getItem(PW_STORAGE);
  if (saved) els.accessPw.value = saved;
} catch {
  /* storage unavailable (private mode etc.) — password just won't persist */
}

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

