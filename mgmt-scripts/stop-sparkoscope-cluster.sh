#!/bin/bash 

USERNAME=hduser
HOSTS="discus-p2irc-mario discus-p2irc-luigi discus-p2irc-worker1 discus-p2irc-worker2 discus-p2irc-worker3 discus-p2irc-worker4 discus-p2irc-worker5 discus-p2irc-worker6 discus-p2irc-worker7 discus-p2irc-worker8 discus-p2irc-worker9"

echo "**********Stopping Flink**********"

cd /usr/local/flink/
bin/stop-cluster.sh
cd

echo "**********Flink Stopped Successfully**********"

echo "**********Stopping Spark**********"

cd /usr/local/sparkoscope/
sbin/stop-slaves.sh
sbin/stop-master.sh
sbin/stop-history-server.sh
cd

echo "**********Spark Stopped Successfully**********"

echo "**********Stopping hadoop**********"

cd /usr/local/hadoop/
sbin/stop-yarn.sh
sbin/stop-dfs.sh
sbin/mr-jobhistory-daemon.sh stop historyserver
cd

echo "**********hadoop Stopped Successfully**********"

echo "**********Stopping NFS on master**********"

sudo service rpcbind stop

cd /usr/local/hadoop/
./sbin/hadoop-daemon.sh --script ./bin/hdfs stop nfs3
./sbin/hadoop-daemon.sh --script ./bin/hdfs stop portmap
sudo umount.nfs /data/mounted_hdfs_path/ -l

echo "**********NFS Successfully Setup on master**********"

SCRIPT="sudo service rpcbind stop; cd /usr/local/hadoop/; ./sbin/hadoop-daemon.sh --script ./bin/hdfs stop nfs3; ./sbin/hadoop-daemon.sh --script ./bin/hdfs stop portmap; sudo umount.nfs /data/mounted_hdfs_path/ -l; cd"

echo "**********Executing commands to stop NFS on all the slaves**********"

for HOSTNAME in ${HOSTS} ; do
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"
    echo "**********NFS stopped sucessfully on " ${HOSTNAME} "**********" 
done

echo ""
echo "==============================="
echo "SUCCESS: Cluster is now down"
echo "==============================="
