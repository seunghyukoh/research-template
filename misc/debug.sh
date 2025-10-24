#!/bin/bash

SRC=./
DST=/<some_remote_path>/debug/

PORT=22
USER=<your_remote_user>
HOST=<your_remote_host>


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
