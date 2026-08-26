// Encrypts the demo API key with the shared access passphrase so the GitHub
// Pages bundle ships only ciphertext. The browser (pages/app.js) reverses this
// with WebCrypto: PBKDF2-SHA256 -> AES-256-GCM. Run by the deploy workflow:
//   DEMO_OPENAI_API_KEY=sk-... DEMO_PASSPHRASE='...' node pages/encrypt_key.mjs <out.json>
import { webcrypto as crypto } from "node:crypto";
import { writeFileSync } from "node:fs";

const apiKey = process.env.DEMO_OPENAI_API_KEY;
const passphrase = process.env.DEMO_PASSPHRASE;
const outPath = process.argv[2] || "key.enc.json";

const missing = [
  !apiKey && "DEMO_OPENAI_API_KEY",
  !passphrase && "DEMO_PASSPHRASE",
].filter(Boolean);
if (missing.length) {
  console.error(
    `Missing env: ${missing.join(", ")}. Add them under Settings -> Secrets and variables -> ` +
      "Actions -> Repository secrets (the Secrets tab, not Variables), then re-run the workflow.",
  );
  process.exit(1);
}

const ITERATIONS = 600000;
const salt = crypto.getRandomValues(new Uint8Array(16));
const iv = crypto.getRandomValues(new Uint8Array(12));

const baseKey = await crypto.subtle.importKey(
  "raw",
  new TextEncoder().encode(passphrase),
  "PBKDF2",
  false,
  ["deriveKey"],
);
const aesKey = await crypto.subtle.deriveKey(
  { name: "PBKDF2", salt, iterations: ITERATIONS, hash: "SHA-256" },
  baseKey,
  { name: "AES-GCM", length: 256 },
  false,
  ["encrypt"],
);
const ciphertext = new Uint8Array(
  await crypto.subtle.encrypt({ name: "AES-GCM", iv }, aesKey, new TextEncoder().encode(apiKey)),
);

const b64 = (bytes) => Buffer.from(bytes).toString("base64");
writeFileSync(
  outPath,
  JSON.stringify(
    {
      v: 1,
      kdf: "PBKDF2-SHA256",
      iterations: ITERATIONS,
      salt: b64(salt),
      iv: b64(iv),
      ciphertext: b64(ciphertext),
    },
    null,
    2,
  ) + "\n",
);
console.log(`Wrote ${outPath}`);
