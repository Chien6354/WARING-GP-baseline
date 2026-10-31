#!/bin/bash

start_num=1
end_num=30

for i in $(seq $start_num $end_num); do
    log_file="log/main_${i}.log"
    pkl_file="log/latest_population_${i}.pkl"

    command="python main.py --log_file $log_file --pkl_file $pkl_file"

    echo "Running: $command"
    $command
done
