# From tensorflow/models/research/
#INPUT_TYPE=image_tensor
#PIPELINE_CONFIG_PATH={RealTimeObjectDetection/Tensorflow/workspace/models/my_ssd_mobnet_new/pipeline.config}
#TRAINED_CKPT_PREFIX={RealTimeObjectDetection/Tensorflow/workspace/models/my_ssd_mobnet_new/ckpt-9.index}
#EXPORT_DIR={final_model}
#python exporter_scripts/export_inference_graph.py \
#    --input_type=${INPUT_TYPE} \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
#    --output_directory=${EXPORT_DIR}
output_directory='inference_graph'
model_dir='/home/raka/Workspace/smartcity/machine-learning/RealTimeObjectDetection/Tensorflow/workspace/models/my_ssd_mobnet_new'
pipeline_config_path='/home/raka/Workspace/smartcity/machine-learning/RealTimeObjectDetection/Tensorflow/workspace/models/my_ssd_mobnet_new/pipeline.config'

python exporter_scripts/exporter_main_v2.py \
    --trained_checkpoint_dir $model_dir \
    --output_directory $output_directory \
    --pipeline_config_path $pipeline_config_path
