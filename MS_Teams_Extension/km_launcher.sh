#!/bin/zsh
set -euo pipefail

ENV_NAME="ehss"
APP_PATH="$HOME/Desktop/GITHUB/EHSS_Hazardous_Waste_Automation/app.py"
PORT=8501

# Load conda into this shell
if [ -x "$HOME/miniconda3/bin/conda" ]; then
  eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
elif [ -x "$HOME/anaconda3/bin/conda" ]; then
  eval "$($HOME/anaconda3/bin/conda shell.zsh hook)"
fi

conda activate "$ENV_NAME"

# If Streamlit already listening, just open the browser; else start it
if lsof -iTCP:$PORT -sTCP:LISTEN -n >/dev/null 2>&1; then
  :
else
  # Start Streamlit in background, log to a file, return immediately
  nohup streamlit run "$APP_PATH" --server.port $PORT > /tmp/km_streamlit.log 2>&1 &
  # tiny grace period so the HTTP server starts
  sleep 2
fi

# Open in Safari (or default browser if you prefer: `open http://localhost:$PORT`)
open -a "Safari" "http://localhost:$PORT"

