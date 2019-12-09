python /home/amgada/projects/def-hefeeda/amgada/cmpt726_project/examples/run_multiple_choice.py \
--model_type bert \
--task_name movieqa \
--model_name_or_path output_albert_90_8 \
--do_eval \
--do_lower_case \
--data_dir /home/amgada/projects/def-hefeeda/amgada/cmpt726_project/datasets/MovieQA/data \
--max_seq_length 80 \
--output_dir /home/amgada/projects/def-hefeeda/amgada/cmpt726_project/output_albert_90_8_test/ \
--per_gpu_eval_batch_size=4 \
--overwrite_output

