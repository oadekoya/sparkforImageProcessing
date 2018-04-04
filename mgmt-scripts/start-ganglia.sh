#!/bin/bash 

USERNAME=hduser
HOSTS="discus-p2irc-mario discus-p2irc-luigi discus-p2irc-worker1 discus-p2irc-worker2 discus-p2irc-worker3 discus-p2irc-worker4 discus-p2irc-worker5 discus-p2irc-worker6 discus-p2irc-worker7 discus-p2irc-worker8 discus-p2irc-worker9"

echo "Starting ganglia on master node"

sudo service ganglia-monitor restart
sudo service gmetad restart

echo "**********Ganglia restarted sucessfully on master**********"

echo "Restarting ganglia on all nodes"

SCRIPT="sudo service ganglia-monitor restart"

for HOSTNAME in ${HOSTS} ; do
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"
    echo "**********Ganglia restarted sucessfully on " ${HOSTNAME} "**********" 
done


echo "Ganglia restarted successfully on all nodes"



