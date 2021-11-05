import os

def main():
    datasets = ["RTE", "MRPC", "STS-B", "CoLA"]
    models = ["not_debiased", "standard_debiased", "reinit_debiased"]

    for seed in range(25, 50):
        for dataset in datasets:
            for model in models:
                if model == "not_debiased":
                    reinit = ""
                    optimizer = "--use_bertadam"
                elif model == "standard_debiased":
                    reinit = ""
                    optimizer = "--use_torch_adamw"
                elif model == "reinit_debiased":
                    reinit = "--reinit_pooler --reinit_layers 5 "
                    optimizer = "--use_torch_adamw"
                
                output_path = "bert_output/"+ str(model) +"/"+ str(dataset) +"/SEED"+ str(seed)

                bash_str = ""\
                + "python run_glue.py "\
                + "--model_type bert --model_name_or_path bert-large-uncased --task_name "+ str(dataset) +" "\
                + "--do_train --data_dir /home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/glue/glue_data/"+ str(dataset) +" --max_seq_length 128 "\
                + "--per_gpu_eval_batch_size 64 --weight_decay 0 --seed "+ str(seed) +" --fp16 "\
                + "--overwrite_output_dir --do_lower_case --per_gpu_train_batch_size 32 "\
                + "--gradient_accumulation_steps 1 --logging_steps 0 --num_loggings 10 "\
                + "--save_steps 0 --test_val_split "+ str(optimizer) +" --cache_dir /home/jovyan/working/class_projects/nlp_11711_project/bert_finetuning_test/cache "\
                + "--num_train_epochs 3.0 --warmup_ratio 0.1 --learning_rate 2e-05 "\
                + "--output_dir " + output_path + " "\
                + reinit\

                os.chdir("/home/jovyan/working/class_projects/nlp_11711_project/revisit-bert-finetuning/replicate")
                os.system(bash_str)



if __name__ == "__main__":
    main()