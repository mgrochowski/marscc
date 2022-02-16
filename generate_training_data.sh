#!/bin/bash
# Generate training data 
# Input: Mars images 1:500K scale

input_path='data/mars_images_scale_1_500K/'
output_path='data/mars_data_220216/'

output_width=480
output_height=480
overlap=240

train_files="${input_path}/train.txt"
val_files="${input_path}/val.txt"
test_files="${input_path}/test.txt"

function run
{
   while read image mask 
   do
      echo ${image} mask:${mask}
      input_dir=$(dirname ${input_files})
      input_file=${input_dir}/${image}
      [ ! -e "${input_file}" ] && { echo "File ${input_file} missing"; exit 1; }
      mask_file=${input_dir}/${mask}
      [ ! -e "${mask_file}" ] && { echo "File ${mask_file} missing"; exit 1; }
   
      python generate_training_data.py --input_file "${input_file}" \
                                       --mask_file "${mask_file}" \
                                       --output_dir "${output_dir}" \
                                       --output_height ${output_height} \
                                       --output_width ${output_width} \
                                       --resize_ratio ${resize_ratio} \
                                       --overlap ${overlap} --debug
   done <  ${input_files}
} 

##############  TRAIN DATA ##################
echo "Generate training data"
resize_ratio="0.1"
output_dir="${output_path}/train"
input_files="${train_files}"
[ ! -f "${input_files}" ] && {  echo "ERROR: Missing ${input_files}"; exit 1; }
run

# 
resize_ratio="0.05"
output_dir="${output_path}/train"
input_files="${train_files}"
run

##############  VAL DATA ##################
echo "Generate valiadtion data"
resize_ratio="0.1"
output_dir="${output_path}/val"
input_files="${val_files}"
[ ! -f "${input_files}" ] && {  echo "ERROR: Missing ${input_files}"; exit 1; }
run

resize_ratio="0.05"
output_dir="${output_path}/val"
input_files="${val_files}"
run

# 
##############  TEST DATA ##################
echo "Generate test data"
resize_ratio="0.1"
output_dir="${output_path}/test_0.1"
input_files="${test_files}"
[ ! -f "${input_files}" ] && {  echo "ERROR: Missing ${input_files}"; exit 1; }
run

# 
resize_ratio="0.05"
output_dir="${output_path}/test_0.05"
input_files="${test_files}"
run

