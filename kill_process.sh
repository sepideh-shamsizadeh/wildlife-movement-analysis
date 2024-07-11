#!/bin/bash

PORT=8765
PID=$(sudo lsof -t -i:$PORT)

if [ -n "$PID" ]; then
  echo "Killing process $PID using port $PORT"
  sudo kill -9 $PID
else
  echo "No process using port $PORT"
fi
