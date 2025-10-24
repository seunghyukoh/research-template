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
