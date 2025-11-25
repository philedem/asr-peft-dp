#!/bin/bash
# Deploy RTSP Transcription API and Frontend to Spark2

set -e

echo "================================================"
echo "RTSP Live Transcription Deployment"
echo "================================================"

# Configuration
REGISTRY="intmidlaiercscr71ee46dc.azurecr.io"
API_IMAGE="${REGISTRY}/rtsp-transcription-api:latest"
FRONTEND_IMAGE="${REGISTRY}/asr-peft-frontend-live:latest"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: Must run from project root"
    exit 1
fi

echo ""
echo "Step 1: Build and push API backend..."
echo "--------------------------------------"
cd backend
docker build -f Dockerfile.rtsp-api -t ${API_IMAGE} .
docker push ${API_IMAGE}
echo "✓ API image pushed"

cd ..

echo ""
echo "Step 2: Build and push frontend..."
echo "--------------------------------------"
cd frontend

# Copy the live environment config
cp .env.live .env

# Build frontend
npm run build

# Create frontend Dockerfile if not exists
if [ ! -f "Dockerfile.live" ]; then
    cat > Dockerfile.live <<'EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy built app
COPY build ./build
COPY .env.live ./.env

# Install serve or use adapter-node
RUN npm install -g serve

EXPOSE 3000

CMD ["serve", "-s", "build", "-l", "3000"]
EOF
fi

docker build -f Dockerfile.live -t ${FRONTEND_IMAGE} .
docker push ${FRONTEND_IMAGE}
echo "✓ Frontend image pushed"

cd ..

echo ""
echo "Step 3: Deploy to Spark2..."
echo "--------------------------------------"

# Apply API deployment
kubectl apply -f k8s/rtsp-transcription-api.yaml
echo "✓ API deployment applied"

# Create frontend deployment if not exists
if [ ! -f "k8s/rtsp-frontend-live.yaml" ]; then
    cat > k8s/rtsp-frontend-live.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtsp-frontend-live
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rtsp-frontend-live
  template:
    metadata:
      labels:
        app: rtsp-frontend-live
    spec:
      imagePullSecrets:
      - name: asr-peft-pull-secret
      containers:
      - name: frontend
        image: ${FRONTEND_IMAGE}
        imagePullPolicy: Always
        ports:
        - containerPort: 3000
          name: http
        env:
        - name: VITE_RTSP_BACKEND_URL
          value: "http://rtsp-transcription-api:8000"
        resources:
          limits:
            memory: "256Mi"
            cpu: "200m"
          requests:
            memory: "128Mi"
            cpu: "100m"
---
apiVersion: v1
kind: Service
metadata:
  name: rtsp-frontend-live
  namespace: default
spec:
  selector:
    app: rtsp-frontend-live
  ports:
  - port: 3000
    targetPort: 3000
    name: http
    nodePort: 30001
  type: NodePort
EOF
fi

kubectl apply -f k8s/rtsp-frontend-live.yaml
echo "✓ Frontend deployment applied"

echo ""
echo "Step 4: Wait for pods to be ready..."
echo "--------------------------------------"
kubectl wait --for=condition=ready pod -l app=rtsp-transcription-api --timeout=60s || true
kubectl wait --for=condition=ready pod -l app=rtsp-frontend-live --timeout=60s || true

echo ""
echo "================================================"
echo "Deployment Summary"
echo "================================================"
echo "API Backend:"
kubectl get pods -l app=rtsp-transcription-api
echo ""
echo "Frontend:"
kubectl get pods -l app=rtsp-frontend-live
echo ""
echo "Services:"
kubectl get svc rtsp-transcription-api rtsp-frontend-live
echo ""
echo "PoC Backend:"
kubectl get pods -l app=whisperlive-rtsp-poc
echo ""
echo "================================================"
echo "✓ Deployment complete!"
echo ""
echo "Access frontend at: http://192.168.4.11:30001/live"
echo "API health check: http://192.168.4.11:30000/health"
echo ""
echo "To view logs:"
echo "  kubectl logs -f -l app=rtsp-transcription-api"
echo "  kubectl logs -f -l app=rtsp-frontend-live"
echo "  kubectl logs -f -l app=whisperlive-rtsp-poc"
echo "================================================"
