#!/bin/bash

job_name_prefix_flowerCounter='flowerCounter_'

input_path_prefix="hdfs://discus-p2irc-master:54310/user/hduser/plot_images"


flowerCount_job_counter=0

ui_port=4039

plot_images=()

for images in /data/mounted_hdfs_path/user/hduser/plot_images/*; do
    plot_images+=("$images")

done



images_length=${#plot_images[@]}



#echo $images_length


for i in {0..4} 
do
    #Inter-arrival rate parameter(lambda) = 1/60 ~= 0.016. This means that on average
    #a new job will be submitted after every 60 seconds (or 1 minute)
    poisson_inter_arrival=$(python -c "import random;print(int(random.expovariate(0.016)))")

    ui_port=$((ui_port+1))


    job_name_flowerCounter=$job_name_prefix_flowerCounter$"job_"$flowerCount_job_counter
    echo "Submitting ---> flowerCounter_job_"$job_name_flowerCounter "then sleep for" $poisson_inter_arrival "seconds";

    # Submit jobs to spark standalone cluster manager

    #spark-submit --master spark://discus-p2irc-master:7077 /data/scripts/flowerCounter.py ${ARRAY[i]} & sleep $poisson_inter_arrival;

    image_path=${plot_images[i]}

    input_path="$input_path_prefix/${image_path:48:63}"

    # Submit jobs to Mesos cluster manager
    
    spark-submit --master spark://discus-p2irc-master:7077 --conf spark.ui.port=$ui_port /data/scripts/flowerCounter.py $input_path $image_path $job_name_flowerCounter & sleep $poisson_inter_arrival;

    #spark-submit --master spark://discus-p2irc-master:7077 --conf spark.ui.port=$ui_port /data/scripts/flowerCounter.py $input_path $image_path $job_name_flowerCounter & sleep 60;

    flowerCount_job_counter=$((flowerCount_job_counter+1))

    # Submit jobs to YARN cluster manager

    #spark-submit --master yarn /data/scripts/flowerCounter.py ${ARRAY[i]} & sleep $poisson_inter_arrival;


done

echo "Jobs have been submitted"    
