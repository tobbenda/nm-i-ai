# GCloud GPU Workflow

## Setup (one-time)

### Prerequisites
- `gcloud` CLI installed (`brew install google-cloud-sdk`)
- Authenticated: `gcloud auth login`
- Project: `nm-ai-prep-2026`

### VM details
```
Name:     ml-rag-vm
Zone:     europe-west4-a
GPU:      NVIDIA L4 (24GB VRAM)
Type:     Spot (~$0.40/hr)
OS:       Ubuntu 22.04 + PyTorch 2.7 + CUDA 12.8
Auto-off: 4 hours after boot
```

### Budget alert
$50 budget alert configured on billing account. Alerts at 50%, 90%, 100%. **Alerts are notifications only — they do not stop the VM.**

---

## Daily workflow

### 1. Start the VM
```bash
gcloud compute instances start ml-rag-vm \
  --zone=europe-west4-a --project=nm-ai-prep-2026
```

### 2. SSH in
```bash
gcloud compute ssh ml-rag-vm \
  --zone=europe-west4-a --project=nm-ai-prep-2026
```

### 3. Pull latest code
```bash
cd ~/nm-ai-prep-26
git pull
cd DM-i-AI-2025/emergency-healthcare-rag
```

### 4. Start vLLM (if not already running)
```bash
# Check if already running
curl -s http://localhost:8081/health

# If not, start it (Llama or Qwen):
vllm serve meta-llama/Llama-3.2-3B-Instruct --max-model-len 8192 --port 8081
# Or: vllm serve Qwen/Qwen2.5-3B-Instruct --max-model-len 8192 --port 8081

# Detach: Ctrl+A, D (if using screen), or run with nohup:
nohup vllm serve meta-llama/Llama-3.2-3B-Instruct --max-model-len 8192 --port 8081 > /tmp/vllm.log 2>&1 &
```

### 5. Run experiments
```bash
USE_VLLM=1 python3 -c "from model import predict; print(predict('test statement'))"

# Or run the evaluator:
USE_VLLM=1 python3 evaluate.py
```

### 6. Stop the VM when done
```bash
# From your laptop (or from the VM itself):
gcloud compute instances stop ml-rag-vm \
  --zone=europe-west4-a --project=nm-ai-prep-2026
```

---

## Iteration cycle

```
 LOCAL (laptop)                         CLOUD (ml-rag-vm)
 ─────────────                          ──────────────────
 1. Edit code in VS Code/Claude
 2. git commit && git push
                    ───────────────▶
                                        3. git pull
                                        4. USE_VLLM=1 python3 evaluate.py
                                        5. Check results
                    ◀───────────────
 6. Analyze, tweak, repeat
```

**Typical cycle time:** ~1-2 min (push + pull + run)

---

## Quick reference

```bash
# Start VM
gcloud compute instances start ml-rag-vm --zone=europe-west4-a --project=nm-ai-prep-2026

# Stop VM (SAVES MONEY)
gcloud compute instances stop ml-rag-vm --zone=europe-west4-a --project=nm-ai-prep-2026

# SSH
gcloud compute ssh ml-rag-vm --zone=europe-west4-a --project=nm-ai-prep-2026

# Check VM status
gcloud compute instances describe ml-rag-vm --zone=europe-west4-a --project=nm-ai-prep-2026 --format="value(status)"

# Check GPU on VM
nvidia-smi

# Check vLLM health
curl -s http://localhost:8081/health

# Tail vLLM logs
tail -f /tmp/vllm.log

# Emergency: copy file directly (bypass git)
gcloud compute scp ./model.py ml-rag-vm:~/nm-ai-prep-26/DM-i-AI-2025/emergency-healthcare-rag/ \
  --zone=europe-west4-a --project=nm-ai-prep-2026
```

---

## Cost notes

| Resource | Cost | When |
|----------|------|------|
| L4 Spot VM running | ~$0.40/hr | While RUNNING |
| VM stopped (disk only) | ~$0.01/hr | While STOPPED |
| 4hr session | ~$1.60 | Auto-shutdown kicks in |
| Forgot overnight (12hr) | ~$4.80 | Don't do this |

- VM auto-shuts down after 4 hours per boot
- Spot VMs can be preempted (killed) by Google anytime — disk survives
- Ollama service auto-starts on boot, vLLM does not (start manually)

---

## Switching between Ollama and vLLM

| | Ollama | vLLM |
|---|--------|------|
| Start | `sudo systemctl start ollama` | `vllm serve ... --port 8081` |
| Speed | Baseline | ~2.4x faster |
| Model format | GGUF (quantized, ~2GB) | HF (FP16, ~6GB) |
| Env var | `USE_VLLM=0` (default) | `USE_VLLM=1` |
| GPU usage | Moderate | Maximized |

---

## Hosting a public API

For competitions that call your API endpoint (e.g. DM i AI), you need a publicly accessible server.

### Option 1: Direct IP (fastest, recommended)

Open firewall once:
```bash
gcloud compute firewall-rules create allow-api-port \
  --allow=tcp:4242 --source-ranges=0.0.0.0/0 \
  --project=nm-ai-prep-2026
```

Start API on VM:
```bash
cd ~/your-project
nohup env HOST_IP=0.0.0.0 CONTAINER_PORT=4242 python3 api.py > /tmp/api.log 2>&1 &
```

Get external IP:
```bash
gcloud compute instances describe ml-rag-vm --zone=europe-west4-a \
  --project=nm-ai-prep-2026 --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

Submit with: `http://<EXTERNAL_IP>:4242`

**Pros:** Lowest latency (~5-10ms from same region), no extra dependencies.
**Cons:** IP changes on restart (spot VM). No HTTPS.

### Option 2: ngrok tunnel

Install ngrok on VM and tunnel:
```bash
ngrok http 4242
```

**Pros:** Stable HTTPS URL, works through NAT/firewalls.
**Cons:** Adds ~3ms latency per request. Free tier has browser warning (requires `ngrok-skip-browser-warning` header). Can cause SSL errors with Python `requests`.

### Latency comparison

| Setup | Per-request latency | Notes |
|-------|-------------------|-------|
| Direct IP (same region) | ~5-10ms | Best for competitions |
| Direct IP (cross-region) | ~50ms | Laptop in Norway → VM in Netherlands |
| ngrok | ~50-55ms | Similar to direct, +3ms for extra hop |
| ngrok (cold/first request) | ~500ms-2s | Free tier interstitial |

**Key insight:** For competition eval servers, what matters is eval server → your API latency. Deploy in the same cloud region as the eval server.

### Deploy key setup (one-time per repo)

Generate key on VM:
```bash
ssh-keygen -t ed25519 -C 'deploy-key-name' -f ~/.ssh/repo_key -N ''
cat ~/.ssh/repo_key.pub  # Add this as deploy key on GitHub
```

Configure SSH to use it:
```bash
cat >> ~/.ssh/config << 'EOF'
Host github.com
    IdentityFile ~/.ssh/repo_key
    IdentitiesOnly yes
EOF
```

**Note:** GitHub requires unique keys per repo. If "Key is already in use", generate a new one with a different filename.

---

## Troubleshooting

**SSH refused right after start:** SSH daemon takes ~30s to start after boot. Wait and retry. If it persists, the VM may have been preempted.

**Spot VM IP changes on restart:** The external IP is ephemeral. Always re-check it after starting the VM:
```bash
gcloud compute instances describe ml-rag-vm --zone=europe-west4-a \
  --project=nm-ai-prep-2026 --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

**SSH fails:** VM might be OOM or preempted. Check status, reset if needed:
```bash
gcloud compute instances describe ml-rag-vm --zone=europe-west4-a --project=nm-ai-prep-2026 --format="value(status)"
gcloud compute instances reset ml-rag-vm --zone=europe-west4-a --project=nm-ai-prep-2026
```

**vLLM gated model error:** Token needs "public gated repo" permission at https://huggingface.co/settings/tokens

**Out of GPU memory:** Kill competing processes:
```bash
nvidia-smi
kill -9 <PID>
```
