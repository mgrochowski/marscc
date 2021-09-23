# train


train_files='data_input/train.txt'
val_files='data_input/val.txt'
test_files='data_input/test.txt'

output_width=480
output_height=480
overlap=240
resize_ratio="0.1"

output_dir=data_output/train
input_files=${train_files}

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
output_dir=data_output/train
input_files=${train_files}
run

# 
resize_ratio="0.05"
output_dir=data_output/train
input_files=${train_files}
run

##############  VAL DATA ##################
echo "Generate valiadtion data"
resize_ratio="0.1"
output_dir=data_output/val
input_files=${val_files}
run

resize_ratio="0.05"
output_dir=data_output/val
input_files=${val_files}
run

# 
##############  TEST DATA ##################
echo "Generate test data"
resize_ratio="0.1"
output_dir="data_output/test_0.1"
input_files=${test_files}
run

# 
resize_ratio="0.05"
output_dir="data_output/test_0.05"
input_files=${test_files}
run

