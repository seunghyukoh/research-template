#!/bin/bash

SRC=./
DST=${DST:-"~/debug"}

PORT=${PORT:-22}
USER=${USER:-$(whoami)}
HOST=${HOST:-"127.0.0.1"}


function dsync() {
    echo "Syncing debug files to $USER@$HOST:$DST"
    rsync -avzP --progress $SRC -e "ssh -p $PORT" $USER@$HOST:$DST
}


function drun() {
    # Sync files first
    dsync
    
    # Run script remotely
    script_args=$1
    echo "Running debug script on $USER@$HOST"
    ssh -p $PORT $USER@$HOST "cd $DST && $script_args"
}
