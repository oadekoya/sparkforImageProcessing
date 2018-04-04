#!/usr/bin/env bash

#for i in {1..9}
#do
# cat ssh_keys | ssh hduser@discus-p2irc-worker$i "cat > ~/.ssh/authorized_keys"
#done
#cat ssh_keys | ssh hduser@discus-p2irc-mario "cat > ~/.ssh/authorized_keys"
#cat ssh_keys | ssh hduser@discus-p2irc-luigi "cat > ~/.ssh/authorized_keys"
#echo "test"

for i in {1..9}
do
 echo "Inside worker $i"
 #ssh hduser@discus-p2irc-worker$i "sudo pip install hdfs"
 ssh hduser@discus-p2irc-worker$i "cp -rf /data/mounted_hdfs_path/fastlmm/FaST-LMM ~/ && cd ~/FaST-LMM && sudo pip install -e ."
done
#ssh hduser@discus-p2irc-mario "sudo pip install hdfs"
ssh hduser@discus-p2irc-mario "cp -rf /data/mounted_hdfs_path/fastlmm/FaST-LMM ~/ && cd ~/FaST-LMM && sudo pip install -e ."
#ssh hduser@discus-p2irc-luigi "sudo pip install hdfs"
ssh hduser@discus-p2irc-luigi "cp -rf /data/mounted_hdfs_path/fastlmm/FaST-LMM ~/ && cd ~/FaST-LMM && sudo pip install -e ."
#echo "test"
