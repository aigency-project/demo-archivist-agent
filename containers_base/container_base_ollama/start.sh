#!/bin/sh
set -e

# Log function for better output formatting
log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1"
}

# Configuration
# Default model if none specified
DEFAULT_MODEL="gemma3:1b"

# Check if models are provided as parameter
if [ -z "$OLLAMA_MODELS" ]; then
  log "No models specified, using default: $DEFAULT_MODEL"
  MODELS_TO_PULL="$DEFAULT_MODEL"
else
  log "Using specified models: $OLLAMA_MODELS"
  MODELS_TO_PULL="$OLLAMA_MODELS"
fi

MAX_RETRIES=30
RETRY_INTERVAL=5



log "Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for the server to be ready by checking with ollama list
log "Waiting for Ollama server to be ready..."
RETRY_COUNT=0

while ! ollama list > /dev/null 2>&1; do
  RETRY_COUNT=$((RETRY_COUNT + 1))
  
  if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    log "ERROR: Ollama server failed to start after $MAX_RETRIES attempts"
    exit 1
  fi
  
  log "Waiting for Ollama server to be ready... (attempt $RETRY_COUNT/$MAX_RETRIES)"
  sleep $RETRY_INTERVAL
done

log "Ollama server is ready!"

# Pull models from environment variable
log "Models to pull: $MODELS_TO_PULL"

# Split the models string by commas and process each model
OLD_IFS="$IFS"
IFS=','
for MODEL in $MODELS_TO_PULL; do
  if [ -n "$MODEL" ]; then
    log "Pulling model: $MODEL"
    ollama pull "$MODEL"
    
    if [ $? -eq 0 ]; then
      log "Successfully pulled model: $MODEL"
    else
      log "WARNING: Failed to pull model: $MODEL"
      # Continue anyway - don't exit on pull failure
    fi
  fi
done
IFS="$OLD_IFS"

log "Initialization complete!"

# Keep the container running by waiting for the server process
wait $SERVER_PID
