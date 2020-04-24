#!/usr/bin/env bash

set -e

TAG=$1

docker build --pull --build-arg CACHEBUST="$(date +%s)" -t "${TAG}" -f docker/Dockerfile .
docker push "${TAG}"
