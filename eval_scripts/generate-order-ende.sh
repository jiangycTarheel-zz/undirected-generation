#!/bin/bash
# Various scripts for generating from models with different algorithms

best_deen_model="../models/best-valid_de-en_mt_bleu.pth"
best_ende_model="../models/best-valid_en-de_mt_bleu.pth"
best_deen_order_model="../models/learned_order_deen_uniform_4gpu/02.10_maxlen80_minlen5_bsz32/tbp1mjg4vi/checkpoint_epoch10+iter34375.pth"
#best_ende_order_model="../models/learned_order_ende_uniform_4gpu/02.11_maxlen80_minlen5_bsz32/n2ccg3b5dw/checkpoint_epoch42+iter134375.pth"
best_ende_order_model="../models/learned_order_ende_uniform_4gpu/02.16_maxlen80_minlen5_bsz32/gn1w0b90zl/checkpoint_epoch68+iter215625.pth"

#checkpoint_epoch70+iter221875.pth
#checkpoint_epoch64+iter203125.pth
#checkpoint_epoch68+iter215625.pth
#checkpoint_epoch72+iter228125.pth

gen_type=${2:-trg2src}
gpuid=${3:-0}
split=${4:-valid}
n_iter=${5:-1}
max_seq_length=30

if [ $gen_type = src2trg ];
  then
    model_path=$best_deen_model
    order_model_path=$best_deen_order_model
  else
    model_path=$best_ende_model
    order_model_path=$best_ende_order_model
fi

function greedy_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --order_model_path ${order_model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --max_seq_length ${max_seq_length}
}

function beam_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --use_data_length --num_topk_lengths 4 --order_beam=4 --split $split --model_path ${model_path} --order_model_path ${order_model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --max_seq_length ${max_seq_length}
}

function left_right_beam_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --max_seq_length ${max_seq_length}
}

function uniform_greedy_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid} --max_seq_length ${max_seq_length}
}

function uniform_beam_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_greedy_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_beam_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function most_least_greedy_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta -1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function most_least_beam_1iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta -1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_beam_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function hard_first_greedy_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha -1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function hard_first_beam_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha -1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_notsimple_beam_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore -m pdb ../adaptive_gibbs_sampler_beam.py --alpha 1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 1 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_greedy_2iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_greedy_4iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 4 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_greedy_2iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_greedy_2iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_2iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_4iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_beam_2iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_beam_2iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_beam_2iter() {
  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_beam_2iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python3 -W ignore ../adaptive_gibbs_sampler_beam_simple.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_greedy_variter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_greedy_variter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_greedy_variter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_variter() {
  python3 -W ignore ../adaptive_gibbs_sampler_simple_with_predicted_order.py --alpha 1.0 --beta 0.9 --gamma 0.0 --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

if [ $1 == "gibbs" ]; then
    gibbs
elif  [ $1 == "greedy_1iter" ]; then
    greedy_1iter
elif  [ $1 == "beam_1iter" ]; then
    beam_1iter
elif  [ $1 == "left_right_greedy_1iter" ]; then
    left_right_greedy_1iter
elif  [ $1 == "left_right_beam_1iter" ]; then
    left_right_beam_1iter
elif  [ $1 == "uniform_greedy_1iter" ]; then
    uniform_greedy_1iter
elif  [ $1 == "uniform_beam_1iter" ]; then
    uniform_beam_1iter
elif  [ $1 == "least_most_greedy_1iter" ]; then
    least_most_greedy_1iter
elif  [ $1 == "least_most_beam_1iter" ]; then
    least_most_beam_1iter
elif  [ $1 == "most_least_greedy_1iter" ]; then
    most_least_greedy_1iter
elif  [ $1 == "most_least_beam_1iter" ]; then
    most_least_beam_1iter
elif  [ $1 == "easy_first_greedy_1iter" ]; then
    easy_first_greedy_1iter
elif  [ $1 == "easy_first_beam_1iter" ]; then
    easy_first_beam_1iter
elif  [ $1 == "hard_first_greedy_1iter" ]; then
    hard_first_greedy_1iter
elif  [ $1 == "hard_first_beam_1iter" ]; then
    hard_first_beam_1iter
  elif  [ $1 == "easy_first_notsimple_beam_1iter" ]; then
      easy_first_notsimple_beam_1iter
elif  [ $1 == "left_right_greedy_2iter" ]; then
    left_right_greedy_2iter
elif  [ $1 == "left_right_greedy_4iter" ]; then
    left_right_greedy_4iter
elif  [ $1 == "uniform_greedy_2iter" ]; then
    uniform_greedy_2iter
elif  [ $1 == "least_most_greedy_2iter" ]; then
    least_most_greedy_2iter
elif  [ $1 == "easy_first_greedy_2iter" ]; then
    easy_first_greedy_2iter
  elif  [ $1 == "easy_first_greedy_4iter" ]; then
      easy_first_greedy_4iter

elif  [ $1 == "left_right_beam_2iter" ]; then
    left_right_beam_2iter
elif  [ $1 == "uniform_beam_2iter" ]; then
    uniform_beam_2iter
elif  [ $1 == "least_most_beam_2iter" ]; then
    least_most_beam_2iter
elif  [ $1 == "easy_first_beam_2iter" ]; then
    easy_first_beam_2iter


elif  [ $1 == "left_right_greedy_variter" ]; then
    left_right_greedy_variter
elif  [ $1 == "uniform_greedy_variter" ]; then
    uniform_greedy_variter
elif  [ $1 == "least_most_greedy_variter" ]; then
    least_most_greedy_variter
elif  [ $1 == "easy_first_greedy_variter" ]; then
    easy_first_greedy_variter
fi
