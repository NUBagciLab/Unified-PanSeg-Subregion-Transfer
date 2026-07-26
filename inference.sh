script="inference_batch.py"
model_dir="/data2/pyq6817/PancreasSegToolkit/Model"

input_dir="/data2/pyq6817/OSU_Pancreas/T1_images"
output_dir="/data2/pyq6817/OSU_Pancreas/T1_partseg"
python $script -i $input_dir -o $output_dir -m $model_dir -d 4 &


input_dir="/data2/pyq6817/OSU_Pancreas/T2_images"
output_dir="/data2/pyq6817/OSU_Pancreas/T2_partseg"
python $script -i $input_dir -o $output_dir -m $model_dir -d 6 &
