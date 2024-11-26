#!/bin/bash

# Check if the process ID is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <process_id>"
  exit 1
fi

process_id=$1
elapsed_time=0

# Loop to check if the process is still running
while kill -0 $process_id 2>/dev/null; do
  echo "Process $process_id is still running... Elapsed time: $elapsed_time seconds"
  sleep 60  # wait for 60 seconds before checking again
  elapsed_time=$((elapsed_time + 60))  # increment elapsed time
done

echo "Process $process_id has stopped after $elapsed_time seconds."
