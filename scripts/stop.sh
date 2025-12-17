# stop qdrant containers (keep as before)
docker ps -aq --filter ancestor=qdrant/qdrant | xargs -r docker rm -f

# chmod +x scripts/stop.sh