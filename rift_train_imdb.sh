work_path="."
mkdir log_bert_imdb

out_path="log_bert_imdb/rs0_clean1_kl10_rinfo0.1"
mkdir $work_path/$out_path
export CUDA_VISIBLE_DEVICES=0
python -u $work_path/rift_train_imdb.py --torch_seed 0 --synonyms_from_file false --genetic_test_num 500 --weight_mi_giveny_adv 0.1  --weight_clean 1 --weight_kl 10  --learning_rate 2e-5 --dataset imdb  --batch_size 32 --test_batch_size 32 --work_path $work_path  --out_path $out_path 

