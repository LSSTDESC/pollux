#!/bin/bash

original_script="run_hacc150_3_subsampledensitymax_method1.sh"

script_content=$(cat $original_script)

for i in 1 2 3 4; do
  for j in 0 1 2 3 4; do
    new_script="run_hacc150_3_subsampledensitymax_method1_${i}_${j}.sh"
    
    new_content=$(echo "$script_content" | sed "s/ ii / ${i} /" | sed "s/ jj / ${j} /")
    
    echo "$new_content" > $new_script
    
    sh $new_script
    
  done
done


original_script="run_hacc195_2_subsampledensitymax_method1.sh"

script_content=$(cat $original_script)

for i in 1 2 3 4; do
  for j in 0 1 2 3 4; do
    new_script="run_hacc195_2_subsampledensitymax_method1_${i}_${j}.sh"
    
    new_content=$(echo "$script_content" | sed "s/ ii / ${i} /" | sed "s/ jj / ${j} /")
    
    echo "$new_content" > $new_script
    
    sh $new_script
    
  done
done


echo "Scripts generated and run."
