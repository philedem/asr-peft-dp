#!/bin/bash

# ASR-PEFT Helm Deployment Script
set -e

NAMESPACE="${NAMESPACE:-default}"
RELEASE_NAME="${RELEASE_NAME:-asr-peft}"
CHART_PATH="./helm/asr-peft"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config.yaml}"

export KUBECONFIG

echo "=========================================="
echo "ASR-PEFT Helm Deployment"
echo "=========================================="
echo "Release: $RELEASE_NAME"
echo "Namespace: $NAMESPACE"
echo "Chart: $CHART_PATH"
echo ""

# Check if Helm is installed
if ! command -v helm &> /dev/null; then
    echo "❌ Helm is not installed. Please install Helm 3.x first."
    exit 1
fi

# Check if kubectl can connect
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster. Check your KUBECONFIG."
    exit 1
fi

echo "✓ Helm and kubectl are ready"
echo ""

# Check if release already exists
if helm list -n "$NAMESPACE" | grep -q "^$RELEASE_NAME"; then
    echo "📦 Release '$RELEASE_NAME' already exists. Upgrading..."
    helm upgrade "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --wait \
        --timeout 5m \
        "$@"
    echo "✅ Upgrade complete!"
else
    echo "📦 Installing release '$RELEASE_NAME'..."
    helm install "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --wait \
        --timeout 5m \
        "$@"
    echo "✅ Installation complete!"
fi

echo ""
echo "=========================================="
echo "Deployment Status"
echo "=========================================="
echo ""

# Show release status
helm status "$RELEASE_NAME" -n "$NAMESPACE"

echo ""
echo "=========================================="
echo "Pod Status"
echo "=========================================="
kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=asr-peft

echo ""
echo "=========================================="
echo "Services & Ingress"
echo "=========================================="
kubectl get svc,ingress -n "$NAMESPACE" | grep asr-peft

echo ""
echo "=========================================="
echo "Access URLs"
echo "=========================================="
FRONTEND_HOST=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.name=="asr-peft-frontend-ingress")].spec.rules[0].host}')
BACKEND_HOST=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.name=="asr-peft-backend-ingress")].spec.rules[0].host}')

echo "Frontend:  https://$FRONTEND_HOST"
echo "Backend:   https://$BACKEND_HOST"
echo ""
echo "🎤 Your ASR-PEFT application is ready!"
