#!/bin/bash

if [ ! -d /data/demo ]; then
    cp -a /usr/local/samples /data/demo
fi

exec /usr/local/bin/nimbix_notebook "$@"
