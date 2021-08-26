#!/bin/bash
# 8 GPUs/node
M5_DEV=lambda-3

LOCAL_PATH=/Users/chaoyanghe/sourcecode/FedGraphNN/

REMOTE_PATH=/home/chaoyanghe/FedGraphNN

# alias ws-sync='rsync -avP -e ssh --delete $LOCAL_PATH hchaoyan@$M5_DEV:$REMOTE_PATH'
alias ws-sync='rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH chaoyanghe@$M5_DEV:$REMOTE_PATH'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done