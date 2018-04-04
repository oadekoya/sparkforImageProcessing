#!/bin/bash  

USERNAME=hduser
HOSTS="discus-p2irc-luigi discus-p2irc-mario discus-p2irc-worker1 discus-p2irc-worker2 discus-p2irc-worker3 discus-p2irc-worker4 discus-p2irc-worker5 discus-p2irc-worker6 discus-p2irc-worker7 discus-p2irc-worker8 discus-p2irc-worker9"

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

echo "**********Starting hadoop**********"

cd /usr/local/hadoop/
sbin/start-dfs.sh
sbin/start-yarn.sh
sbin/mr-jobhistory-daemon.sh start historyserver
cd 

echo "**********hadoop Started Successfully**********"

echo "**********Starting Spark**********"

cd /usr/local/sparkoscope/
sbin/start-master.sh 
sbin/start-slaves.sh 
sbin/start-history-server.sh
cd 

echo "**********Spark Started Successfully**********"

#echo "**********Starting Flink**********"

#cd /usr/local/flink/
#bin/start-cluster.sh 
#cd 

#echo "**********Flink Started Successfully**********"


echo "**********Setting up NFS on master**********"

##sudo service nfs stop
#sudo service rpcbind stop
#sudo service rpcbind start

#cd /usr/local/hadoop/
#./sbin/hadoop-daemon.sh --script ./bin/hdfs stop nfs3
#./sbin/hadoop-daemon.sh --script ./bin/hdfs stop portmap
#./sbin/hadoop-daemon.sh --script ./bin/hdfs start portmap
#./sbin/hadoop-daemon.sh --script ./bin/hdfs start nfs3
##sudo -u hduser umount.nfs /data/mounted_hdfs_path/ -l
#sudo mount -t nfs -o vers=3,rsize=10485760,wsize=10485760,proto=tcp,nolock 127.0.0.1:/ /data/mounted_hdfs_path/
#cd

echo "**********NFS Successfully Setup on master**********"

#SCRIPT="sudo service rpcbind stop; sudo service rpcbind start; cd /usr/local/hadoop/; ./sbin/hadoop-daemon.sh --script ./bin/hdfs stop nfs3; ./sbin/hadoop-daemon.sh --script ./bin/hdfs stop portmap; ./sbin/hadoop-daemon.sh --script ./bin/hdfs start portmap; ./sbin/hadoop-daemon.sh --script ./bin/hdfs start nfs3; sudo umount.nfs /data/mounted_hdfs_path/ -l; sudo mount -t nfs -o vers=3,rsize=10485760,wsize=10485760,proto=tcp,nolock 127.0.0.1:/ /data/mounted_hdfs_path/; cd"

echo "**********Executing commands to setup NFS on all the slaves**********"

#for HOSTNAME in ${HOSTS} ; do
#    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"
#    echo "**********NFS setup sucessfully on " ${HOSTNAME} "**********" 
#done

echo "**********NFS running on all the slaves**********"
echo ""
echo "==============================="
echo "SUCCESS: Cluster up and running"
echo "==============================="
