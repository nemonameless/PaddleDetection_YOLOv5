#python3.7 dygraph_print.py -c configs/mtyolo/mtyolo_t_400e_coco.yml 2>&1 | tee mtyolo_t.txt

export FLAGS_allocator_strategy=auto_growth
name=s
model_type=mtyolo
job_name=mtyolo_${name}_400e_coco
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weight_path=../v6_${name}_coco_paddle.pdparams #https://paddledet.bj.bcebos.com/models/${job_name}.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=6 python3.7 tools/train.py -c ${config}
CUDA_VISIBLE_DEVICES=4 python3.7 tools/eval.py -c ${config} -o weights=${weight_path}
#CUDA_VISIBLE_DEVICES=6 python3.7 tools/infer.py -c ${config} -o weights=${weight_path} --infer_img=demo/000000014439_640x640.jpg
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/export_model.py -c ${config} -o weights=${weight_path}
# 6. deploy speed
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000570688.jpg --device=GPU --run_benchmark=True
