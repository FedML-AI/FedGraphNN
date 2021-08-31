#!/bin/bash
# 8 GPUs/node
M5_DEV=ec2-3-86-4-172.compute-1.amazonaws.com

LOCAL_PATH=/Users/hchaoyan/source/FedGraphNN/

REMOTE_PATH=/home/hchaoyan/FedGraphNN

# alias ws-sync='rsync -avP -e ssh --delete $LOCAL_PATH hchaoyan@$M5_DEV:$REMOTE_PATH'
alias ws-sync='rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH hchaoyan@$M5_DEV:$REMOTE_PATH'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done