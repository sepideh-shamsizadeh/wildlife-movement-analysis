#!/bin/bash

# Number of concurrent connections
NUM_CONNECTIONS=10000

# WebSocket server URL
WS_URL="ws://tool1.example.com"

# Loop to create multiple connections
for i in $(seq 1 $NUM_CONNECTIONS)
do
  websocat $WS_URL &
done

# Wait for all background processes to finish
wait
