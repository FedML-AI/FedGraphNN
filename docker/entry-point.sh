#!/bin/bash
# Launch supervisor
BASENAME="${0##*/}"
log () {
  echo "${BASENAME} - ${1}"
}
AWS_BATCH_EXIT_CODE_FILE="/tmp/batch-exit-code"
supervisord -n -c "/etc/supervisor/supervisord.conf"

log "Reading exit code from batch script stored at $AWS_BATCH_EXIT_CODE_FILE"
if [ ! -f $AWS_BATCH_EXIT_CODE_FILE ]; then
    echo "Exit code file not found , returning with exit code 1!" >&2
    exit 1
fi
exit $(cat $AWS_BATCH_EXIT_CODE_FILE)
