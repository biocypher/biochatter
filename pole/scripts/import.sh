#!/bin/bash -c
sleep 2
ls /data/build2neo/neo4j-admin-import-call.sh
if [ -f /data/build2neo/neo4j-admin-import-call.sh ]; then
  chmod +x /data/build2neo/neo4j-admin-import-call.sh
  /data/build2neo/neo4j-admin-import-call.sh
fi
neo4j start
sleep 10
neo4j stop
