#!/bin/bash
# Various scripts for generating from models with different algorithms

best_deen_model="models/best-valid_de-en_mt_bleu.pth"
best_ende_model="models/best-valid_en-de_mt_bleu.pth"
gen_type=${2:-trg2src}
gpuid=${3:-0}
split=${4:-valid}
n_iter=${5:-1}
max_seq_length=30

if [ $gen_type = src2trg ];
  then
    model_path=$best_deen_model
  else
    model_path=$best_ende_model
fi

function left_right_greedy_1iter() {
  python3 -W ignore adaptive_gibbs_sampler_simple.py --heuristic left2right --alpha 0.0 --beta 0.0 --gamma 1.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --src_lang=${src_lang} --trg_lang=${trg_lang} --data_path=${data_path} --max_seq_length=${max_seq_length}
}

function uniform_greedy_1iter() {
  python3 -W ignore adaptive_gibbs_sampler_simple.py --heuristic uniform --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --src_lang=${src_lang} --trg_lang=${trg_lang} --data_path=${data_path} --max_seq_length=${max_seq_length}
}

function least_most_greedy_1iter() {
  python3 -W ignore adaptive_gibbs_sampler_simple.py --heuristic least_most --alpha 0.0 --beta 1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --src_lang=${src_lang} --trg_lang=${trg_lang} --data_path=${data_path} --max_seq_length=${max_seq_length}
}

function most_least_greedy_1iter() {
  python3 -W ignore adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta -1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --src_lang=${src_lang} --trg_lang=${trg_lang} --data_path=${data_path} --max_seq_length=${max_seq_length}
}

function easy_first_greedy_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore adaptive_gibbs_sampler_simple.py --heuristic easy_first --alpha 1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --src_lang=${src_lang} --trg_lang=${trg_lang} --data_path=${data_path} --max_seq_length=${max_seq_length}
}

function hard_first_greedy_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore adaptive_gibbs_sampler_simple.py --heuristic hard_first --alpha -1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --src_lang=${src_lang} --trg_lang=${trg_lang} --data_path=${data_path}
}


if [ $1 == "gibbs" ]; then
    gibbs
elif  [ $1 == "left_right_greedy_1iter" ]; then
    left_right_greedy_1iter
elif  [ $1 == "uniform_greedy_1iter" ]; then
    uniform_greedy_1iter
elif  [ $1 == "least_most_greedy_1iter" ]; then
    least_most_greedy_1iter
elif  [ $1 == "most_least_greedy_1iter" ]; then
    most_least_greedy_1iter
elif  [ $1 == "easy_first_greedy_1iter" ]; then
    easy_first_greedy_1iter
elif  [ $1 == "hard_first_greedy_1iter" ]; then
    hard_first_greedy_1iter
fi