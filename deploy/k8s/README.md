# Deploying the claim-extraction API to e-INFRA CZ (CERIT-SC Kubernetes)

Target: `https://facticli.dyn.cloud.e-infra.cz` — same platform as FactSearch2.

## 1. Access

Log in to <https://rancher.cloud.e-infra.cz> with e-INFRA CZ federated login and
select your namespace. Download the kubeconfig from the Rancher UI (top right →
*Download KubeConfig*) and point `kubectl` at it.

## 2. Create the secret (never commit these)

```bash
kubectl create secret generic facticli-api-secrets \
  --from-literal=OPENAI_API_KEY='sk-...' \
  --from-literal=OPENAI_API_MODEL='gpt-5.6-terra' \
  --from-literal=OPENAI_API_BASE_URL='https://api.openai.com/v1' \
  --from-literal=FACTICLI_API_KEY='cedmo_2026'
```

## 3. Deploy

```bash
kubectl apply -f deploy/k8s/
kubectl rollout status deployment/facticli-api
```

## 4. Verify

```bash
curl -s https://facticli.dyn.cloud.e-infra.cz/api/health
# {"status":"ok","has_api_key":true,"model_configured":true,"auth_required":true}

curl -s https://facticli.dyn.cloud.e-infra.cz/api/extract \
  -H 'Authorization: Bearer cedmo_2026' -H 'Content-Type: application/json' \
  -d '{"text":"Inflace loni klesla pod 3 procenta.","max_claims":5}'
```

TLS is issued automatically by cert-manager; the first request may take a minute
while the certificate is obtained.

## 5. Point the demo at it

Set the repository **variable** `DEMO_API_BASE` to
`https://facticli.dyn.cloud.e-infra.cz` (Settings → Secrets and variables →
Actions → Variables), then re-run the *Deploy claim extractor demo* workflow.

## Rotating the access key

```bash
kubectl patch secret facticli-api-secrets \
  -p '{"stringData":{"FACTICLI_API_KEY":"new_key"}}'
kubectl rollout restart deployment/facticli-api
```
