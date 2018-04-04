#!/usr/bin/env bash

for i in {1..9}
do
 ssh hduser@discus-p2irc-worker$i "mkdir /tmp/flink"
done
ssh hduser@discus-p2irc-mario "mkdir /tmp/flink"
ssh hduser@discus-p2irc-luigi "mkdir /tmp/flink"
#echo "test"
