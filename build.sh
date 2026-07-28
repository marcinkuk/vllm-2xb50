#!/bin/bash

# 1. Twardy reset do czystego stanu i pobranie najnowszego kodu
docker builder prune -a -f
docker image prune -a -f

git reset --hard origin/main
git pull

HASH=$(git rev-parse --short HEAD)
DATE=$(date +%Y-%m-%d_%H-%M)
NAME=${DATE}-${HASH}

git clean -fdx

curl -L "https://github.com/vllm-project/vllm/compare/main...yma11:vllm:docker-update.patch" -o docker.patch
git apply docker.patch && NAME=${NAME}-p

docker build --cpuset-cpus="0" --memory="16g" --no-cache -f docker/Dockerfile.xpu -t vllm-intel-xpu:${NAME} .
#docker build --cpuset-cpus="0" --memory="8g" -f docker/Dockerfile.xpu -t vllm-intel-xpu:${DATE}-${HASH} .
#docker build --no-cache -f docker/Dockerfile.xpu -t vllm-intel-xpu:${DATE}-${HASH} .

echo vllm-intel-xpu:${NAME}
#git reset --hard origin/main

