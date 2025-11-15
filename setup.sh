#!/bin/bash

# ASR-PEFT-DP Setup Script
# This script initializes the environment and prepares the system for first use

set -e

echo "🚀 ASR-PEFT-DP Setup Script"
echo "============================"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Create environment files if they don't exist
if [ ! -f "backend/.env" ]; then
    echo "📝 Creating backend/.env from template..."
    cp backend/.env.example backend/.env
    echo "✅ Created backend/.env"
else
    echo "⏭️  backend/.env already exists, skipping..."
fi

if [ ! -f "frontend/.env" ]; then
    echo "📝 Creating frontend/.env from template..."
    cp frontend/.env.example frontend/.env
    echo "✅ Created frontend/.env"
else
    echo "⏭️  frontend/.env already exists, skipping..."
fi

echo ""

# Create data directories
echo "📁 Creating data directories..."
mkdir -p backend/data/audio
mkdir -p backend/data/records
mkdir -p backend/data/lora_output
mkdir -p backend/mlruns

# Initialize counter files
echo "0" > backend/data/manual_review_count.txt
echo "WER: Not calculated yet" > backend/data/wer.txt

echo "✅ Data directories created"
echo ""

# Build and start containers
echo "🔨 Building Docker containers (this may take several minutes)..."
docker-compose build

echo ""
echo "✅ Build complete!"
echo ""

# Start services
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Setup complete! Services are running."
    echo ""
    echo "📊 Access points:"
    echo "   Frontend:  http://localhost:5173"
    echo "   Backend:   http://localhost:8000"
    echo "   API Docs:  http://localhost:8000/docs"
    echo ""
    echo "📚 Next steps:"
    echo "   1. Open http://localhost:5173 in your browser"
    echo "   2. Upload or record audio"
    echo "   3. Review and correct transcriptions"
    echo "   4. Model will auto-retrain after 20 corrections"
    echo ""
    echo "🔧 Useful commands:"
    echo "   View logs:        docker-compose logs -f"
    echo "   Stop services:    docker-compose down"
    echo "   Restart:          docker-compose restart"
    echo ""
else
    echo ""
    echo "⚠️  Warning: Services may not have started correctly."
    echo "   Run 'docker-compose logs' to check for errors."
    echo ""
fi
