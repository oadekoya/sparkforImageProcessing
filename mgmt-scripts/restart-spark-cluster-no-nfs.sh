#!/bin/bash  

echo "**********Stopping Spark**********"

cd /usr/local/spark-2.1.0/
sbin/stop-slaves.sh
sbin/stop-master.sh
sbin/stop-history-server.sh
cd

echo "**********Spark Stopped Successfully**********"

echo "**********Stopping YARN**********"

cd /usr/local/hadoop/
sbin/stop-yarn.sh
cd

echo "**********YARN Stopped Successfully**********"

echo "**********Starting YARN**********"

cd /usr/local/hadoop/
sbin/start-yarn.sh
cd

echo "**********YARN Started Successfully**********"

echo "**********Starting Spark**********"

cd /usr/local/spark-2.1.0/
sbin/start-master.sh
sbin/start-slaves.sh
sbin/start-history-server.sh
cd

echo "**********Spark Started Successfully**********"

echo "==============================="
echo "SUCCESS: Cluster up and running"
echo "==============================="



