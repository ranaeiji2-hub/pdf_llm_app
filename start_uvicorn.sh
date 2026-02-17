#!/bin/sh

cd /workspace || exit 1

nohup python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  > /tmp/uvicorn.log 2>&1 &
