#!/bin/bash

echo "**********Stopping Flink**********"

cd /usr/local/flink/
bin/stop-cluster.sh
cd

echo "**********Flink Stopped Successfully**********"

echo "**********Starting Flink**********"

cd /usr/local/flink/
bin/start-cluster.sh 
cd 

echo "**********Flink Started Successfully**********"

echo "==============================="
echo "SUCCESS: Cluster up and running"
echo "==============================="
