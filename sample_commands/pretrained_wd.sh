python run_glue.py \
    --model_type bert --model_name_or_path bert-large-uncased --task_name RTE \
    --do_train --data_dir /home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/glue/glue_data/RTE --max_seq_length 128 \
    --per_gpu_eval_batch_size 64 --weight_decay 0 --seed 0 --fp16 \
    --overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 \
    --save_steps 0 --test_val_split --use_torch_adamw --cache_dir /home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/cache \
    --num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 \
    --output_dir bert_output/ORIGINAL/RTE/SEED0 --prior_weight_decay --weight_decay 1e-3