#!/usr/bin/env bash

echo "🚀 Starting Agentic RAG Chatbot Web App..."
echo ""

if [ -d ".venv" ]; then
    echo "✅ Found virtual environment"
    PYTHON_CMD=".venv/bin/python"
else
    echo "⚠️  No .venv found, using system python3"
    PYTHON_CMD="python3"
fi

echo "Checking Flask installation..."
$PYTHON_CMD -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Flask not found. Installing..."
    $PYTHON_CMD -m pip install flask -q
fi

if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Create it with: echo 'OPENAI_API_KEY=your_key' > .env"
fi

if [ ! -f "templates/index.html" ]; then
    echo "❌ Error: templates/index.html not found!"
    exit 1
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "🌐 Starting web server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   📍 Server will start at: http://localhost:5001"
echo "   🧪 Test page: http://localhost:5001/test"
echo "   ❤️  Health check: http://localhost:5001/health"
echo ""
echo "   ℹ️  Using port 5001 (port 5000 is used by macOS AirPlay Receiver)"
echo ""
echo "   ⚠️  If you see a blank page:"
echo "      1. Check terminal for errors"
echo "      2. Visit http://localhost:5000/test to verify server is running"
echo "      3. Try hard refresh: Ctrl+Shift+R (Mac: Cmd+Shift+R)"
echo "      4. Check browser console (F12) for JavaScript errors"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

$PYTHON_CMD -m app.web
