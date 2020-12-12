#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ncs --model-store model-store --models bert=bert.mar
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
