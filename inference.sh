script="inference_batch.py"
model_dir="/data2/pyq6817/PancreasSegToolkit/Model"

input_dir="/data2/pyq6817/Cyst_segmentation_dev/images_n4bias_harmonized"
output_dir="/data2/pyq6817/Cyst_segmentation_dev/probability_map_pancreas"
python $script -i $input_dir -o $output_dir -m $model_dir -d 4 --save_probability &


input_dir="/data2/pyq6817/Cyst_segmentation_dev/images_n4bias_harmonized"
output_dir="/data2/pyq6817/Cyst_segmentation_dev/segmentation_pancreas"
python $script -i $input_dir -o $output_dir -m $model_dir -d 1 &
