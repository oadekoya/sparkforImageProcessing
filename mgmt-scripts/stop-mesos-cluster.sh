#!/bin/bash  

USERNAME=hduser
HOSTS="discus-p2irc-mario discus-p2irc-luigi discus-p2irc-worker1 discus-p2irc-worker2 discus-p2irc-worker3 discus-p2irc-worker4 discus-p2irc-worker5 discus-p2irc-worker6 discus-p2irc-worker7 discus-p2irc-worker8 discus-p2irc-worker9"


echo "**********Stopping Mesos master*********"
sudo service mesos-slave stop
sudo service zookeeper stop
sudo service mesos-master stop --work_dir=/data/mesos/work --log_dir=/data/mesos/logs
sudo service marathon stop


echo "**********Mesos Master stopped successfully**********"

SCRIPT="sudo service mesos-master stop; sudo service zookeeper stop; sudo service mesos-slave stop --work_dir=/data/mesos/work --log_dir=/data/mesos/logs"

echo "**********Stopping all Mesos slaves**********"

for HOSTNAME in ${HOSTS} ; do
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"
    echo "**********Mesos slave stopped successfully on " ${HOSTNAME} "**********" 
done

echo "**********All Mesos slaves stopped successfully**********"
echo ""
echo "==================================="
echo "SUCCESS: Mesos cluster is shut down"
echo "==================================="

