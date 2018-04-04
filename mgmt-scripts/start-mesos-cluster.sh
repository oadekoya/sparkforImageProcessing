#!/bin/bash  

USERNAME=hduser
HOSTS="discus-p2irc-mario discus-p2irc-luigi discus-p2irc-worker1 discus-p2irc-worker2 discus-p2irc-worker3 discus-p2irc-worker4 discus-p2irc-worker5 discus-p2irc-worker6 discus-p2irc-worker7 discus-p2irc-worker8 discus-p2irc-worker9"

#HOSTS="discus-p2irc-worker1 discus-p2irc-worker3 discus-p2irc-worker4 discus-p2irc-worker5 discus-p2irc-worker6 discus-p2irc-worker7 discus-p2irc-worker8 discus-p2irc-worker9"

echo "**********Starting Mesos master*********"
sudo service mesos-slave stop
sudo service zookeeper restart
#sudo service zookeeper stop
#sudo service zookeeper start
sudo service mesos-master restart 
sudo service marathon restart


echo "**********Mesos Master started successfully**********"

SCRIPT="sudo service mesos-master stop; sudo service zookeeper stop; sudo service mesos-slave restart --work_dir=/data/mesos/work --log_dir=/data/mesos/logs"

echo "**********Starting all Mesos slaves**********"

for HOSTNAME in ${HOSTS} ; do
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"
    echo "**********Mesos slave started successfully on " ${HOSTNAME} "**********" 
done

echo "**********All Mesos slaves started successfully**********"
echo ""
echo "============================================"
echo "SUCCESS: Mesos cluster is now up and running"
echo "============================================"

