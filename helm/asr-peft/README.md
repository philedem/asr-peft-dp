# ASR-PEFT Helm Chart

Helm chart for deploying the ASR Parameter-Efficient Fine-Tuning application on Kubernetes.

## Features

- **Backend**: FastAPI service with Whisper ASR and LoRA fine-tuning
- **Frontend**: Svelte-based UI for audio transcription and annotation
- **GPU Support**: Automatic CUDA detection with NVIDIA runtime
- **HTTPS Ingress**: TLS-enabled ingress with Traefik
- **Persistent Storage**: PVCs for model cache and training data

## Prerequisites

- Kubernetes cluster with GPU support
- Helm 3.x
- NVIDIA device plugin installed
- Traefik ingress controller
- TLS secret (default: `midlaier-tls`)
- Container registry credentials configured as `asr-peft-pull-secret`

## Installation

### Quick Install

```bash
# Install with default values
helm install asr-peft ./helm/asr-peft -n default

# Install with custom values
helm install asr-peft ./helm/asr-peft -n default -f custom-values.yaml
```

### Upgrade

```bash
helm upgrade asr-peft ./helm/asr-peft -n default
```

### Uninstall

```bash
helm uninstall asr-peft -n default
```

## Configuration

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.registry` | Container registry | `intmidlaiercscr71ee46dc.azurecr.io` |
| `backend.resources.requests.nvidia.com/gpu` | GPU allocation | `1` |
| `backend.env.enableDifferentialPrivacy` | Enable DP | `"false"` |
| `persistence.data.size` | Data volume size | `10Gi` |
| `persistence.hfCache.size` | Model cache size | `20Gi` |
| `ingress.hosts.frontend.host` | Frontend hostname | `asr.192.168.3.155.nip.io` |
| `ingress.hosts.backend.host` | Backend hostname | `asr-api.192.168.3.155.nip.io` |
| `backendUrl` | Backend API URL | `https://asr-api.192.168.3.155.nip.io` |

### Example Custom Values

```yaml
# custom-values.yaml
ingress:
  hosts:
    frontend:
      host: asr.mydomain.com
    backend:
      host: asr-api.mydomain.com
  tls:
    - secretName: my-tls-secret
      hosts:
        - asr.mydomain.com
        - asr-api.mydomain.com

backendUrl: "https://asr-api.mydomain.com"

backend:
  replicas: 2
  resources:
    requests:
      nvidia.com/gpu: "2"
```

## Access

After deployment, access the application at:

- **Frontend**: `https://asr.192.168.3.155.nip.io`
- **Backend API**: `https://asr-api.192.168.3.155.nip.io`

## Monitoring

```bash
# Check pod status
kubectl get pods -n default -l app.kubernetes.io/name=asr-peft

# View backend logs
kubectl logs -n default -l app=asr-peft-backend --tail=50

# View frontend logs
kubectl logs -n default -l app=asr-peft-frontend --tail=50

# Check GPU allocation
kubectl describe pod -n default -l app=asr-peft-backend | grep -A 5 "Limits:"
```

## Development

### Template Rendering

```bash
# Render templates without installing
helm template asr-peft ./helm/asr-peft -n default

# Render with custom values
helm template asr-peft ./helm/asr-peft -n default -f custom-values.yaml
```

### Linting

```bash
helm lint ./helm/asr-peft
```

## Troubleshooting

### GPU Not Detected

Ensure the NVIDIA device plugin is running:
```bash
kubectl get pods -n kube-system | grep nvidia-device-plugin
```

### Ingress Not Working

Check Traefik ingress controller:
```bash
kubectl get pods -n kube-system | grep traefik
kubectl get ingress -n default
```

### PVC Not Binding

Check storage class and PVC status:
```bash
kubectl get sc
kubectl get pvc -n default
kubectl describe pvc -n default
```
