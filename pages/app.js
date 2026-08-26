"use strict";

/*
 * Browser-only claim extractor for the GitHub Pages demo.
 * Mirrors CompatibleClaimExtractionAdapter (src/facticli/adapters/openai_provider.py):
 * same skill prompt (fetched from ./extract_claims.md, copied in at deploy time),
 * same user payload, same output contract enforced via a strict JSON schema.
 * Access control: the deploy workflow encrypts the demo API key with the shared
 * passphrase (PBKDF2 + AES-GCM, see pages/encrypt_key.mjs) and ships only the
 * ciphertext (./key.enc.json). Entering the correct password decrypts the key
 * in the browser; a wrong password simply fails to decrypt. The decrypted key
 * is held in memory only and sent only to the configured API endpoint.
 */

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

/* Mirrors ClaimExtractionResult/CheckworthyClaim in core/contracts.py,
 * minus input_text (re-attached client-side to avoid echoing long inputs). */
const RESULT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    detected_language: {
      type: "string",
      description:
        "ISO 639-1 code of the input's dominant language (e.g. 'cs', 'sk', 'pl', 'en'). All generated claim text is written in this language.",
    },
    claims: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          claim_id: { type: "string", description: "Stable identifier for the extracted claim." },
          claim_text: {
            type: "string",
            description:
              "Standalone, decontextualized, atomic factual claim suitable for independent checking.",
          },
          source_fragment: {
            type: "string",
            description: "Short fragment from the input text that directly grounds this claim.",
          },
          checkworthy_reason: {
            type: "string",
            description: "Why this claim is check-worthy (impact, specificity, verifiability).",
          },
        },
        required: ["claim_id", "claim_text", "source_fragment", "checkworthy_reason"],
      },
    },
    coverage_notes: {
      type: "array",
      items: { type: "string" },
      description: "How extraction covers the factual content from the input.",
    },
    excluded_nonfactual: {
      type: "array",
      items: { type: "string" },
      description: "Statements intentionally excluded as non-factual/opinion/rhetoric.",
    },
  },
  required: ["detected_language", "claims", "coverage_notes", "excluded_nonfactual"],
};

const $ = (id) => document.getElementById(id);

const els = {
  input: $("input-text"),
  accessPw: $("access-pw"),
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
let promptPromise = null;
let unlockedKey = null;
let unlockedWith = null;

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

function loadPrompt() {
  if (!promptPromise) {
    promptPromise = fetch("./extract_claims.md").then((resp) => {
      if (!resp.ok) throw new Error("Could not load the extraction prompt.");
      return resp.text();
    });
    promptPromise.catch(() => (promptPromise = null));
  }
  return promptPromise;
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

/* ---------- access control ---------- */

const b64ToBytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

async function decryptApiKey(password) {
  const resp = await fetch("./key.enc.json");
  if (!resp.ok) {
    throw new Error(
      "This deployment has no embedded API credential. The repository secrets are probably not configured.",
    );
  }
  const blob = await resp.json();
  const baseKey = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveKey"],
  );
  const aesKey = await crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: b64ToBytes(blob.salt),
      iterations: blob.iterations,
      hash: "SHA-256",
    },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"],
  );
  try {
    const plain = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: b64ToBytes(blob.iv) },
      aesKey,
      b64ToBytes(blob.ciphertext),
    );
    return new TextDecoder().decode(plain).trim();
  } catch {
    throw new Error("Incorrect access password.");
  }
}

async function unlock(password) {
  if (unlockedKey && unlockedWith === password) return unlockedKey;
  unlockedKey = await decryptApiKey(password);
  unlockedWith = password;
  try {
    localStorage.setItem(PW_STORAGE, password);
  } catch {
    /* storage unavailable — password just won't persist */
  }
  return unlockedKey;
}

/* ---------- extraction ---------- */

async function callExtraction(text, maxClaims) {
  const apiKey = await unlock(els.accessPw.value);
  const model = els.model.value.trim() || "gpt-5.6-terra";
  const baseUrl = (els.baseUrl.value.trim() || "https://api.openai.com/v1").replace(/\/+$/, "");

  const prompt = await loadPrompt();

  const payload = {
    input_text: text,
    requirements: {
      max_claims: maxClaims,
      decontextualized: true,
      atomic_claims: true,
      maximize_checkworthy_coverage: true,
      only_directly_mentioned_facts: true,
      detect_and_report_language: true,
      write_output_in_input_language: true,
      preserve_original_diacritics: true,
    },
  };

  const resp = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: prompt },
        { role: "user", content: JSON.stringify(payload, null, 2) },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "claim_extraction_result",
          strict: true,
          schema: RESULT_SCHEMA,
        },
      },
    }),
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    if (resp.status === 401) throw new Error("The embedded demo key was rejected (HTTP 401). It may have been revoked.");
    if (resp.status === 429) throw new Error("Rate limit or budget exceeded (HTTP 429). The demo key may be exhausted.");
    throw new Error(data.error?.message || `Request failed (HTTP ${resp.status}).`);
  }

  const content = data.choices?.[0]?.message?.content;
  if (!content) throw new Error("The model returned an empty response.");
  const result = JSON.parse(content);
  result.input_text = text;
  return result;
}

async function extract() {
  const text = els.input.value.trim();
  if (!text) {
    showError("Please enter some text to analyze.");
    els.input.focus();
    return;
  }
  if (!els.accessPw.value) {
    showError("Please enter the access password (ask the CEDMO team for it).");
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

loadPrompt().catch(() => {});
