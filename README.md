# undirected-generation-dev
This repo contains the source code of the models described in the following paper 
* *"Learning and Analyzing Generation Order for Undirected Sequence Models"* in Findings of EMNLP, 2021. ([paper](https://arxiv.org/abs/2112.09097)).

The basic code structure was adapted from the NYU [dl4mt-seqgen](https://github.com/nyu-dl/dl4mt-seqgen).
We also use the pybleu from [fairseq](https://github.com/pytorch/fairseq) to calculate BLEU scores during the reinforcement learning.

## 0. Preparation
### 0.1 Dependencies
* PyTorch 1.4.0/1.6.0/1.8.0

### 0.2 Data
The WMT'14 De-En data and the pretrained De-En MLM model are provided in the [dl4mt-seqgen](https://github.com/nyu-dl/dl4mt-seqgen).
* Download WMT'14 De-En [*valid/test* data](https://drive.google.com/file/d/11hZN2bctJsGOBUx6of5en4eZw-A66mQs/view?usp=sharing). 
* Then organize the data in `data/` and make sure it follows such a structure:
```
------ data
--------- de-en
------------ train.de-en.de.pth
------------ train.de-en.en.pth
------------ valid.de-en.de.pth
------------ valid.de-en.en.pth
------------ test.de-en.de.pth
------------ test.de-en.en.pth
```

* Download [pretrained models](https://drive.google.com/open?id=1m1R7JC7tSnx3gog-UmeWamEERVoyWfMv).
* Then organize the pretrained masked language models in `models/` make sure it follows such a structure:
```
------ models
--------- best-valid_en-de_mt_bleu.pth
--------- best-valid_de-en_mt_bleu.pth
```

## 2. Training the order policy network with reinforcement learning
Train a policy network to predict the generation order for a pretrained De-En masked language model:
```
./train_scripts/train_order_rl_deen.sh
```
* By defaults, the model checkpoints will be saved in `models/learned_order_deen_uniform_4gpu/00_maxlen30_minlen5_bsz32`.
* By using this script, we are only training the model on De-En sentence pairs where both the German and English sentences
with a maximum length of 30 and a minimum length of 5. You can change the training parameters `max_len` and `min_len` 
to change the length limits.

## 3. Decode the undirected generation model with learned orders
* Set the `MODEL_CKPT` parameter to the corresponding path found under `models/00_maxlen30_minlen5_bsz32`. For example: 
```
export MODEL_CKPT=wj8oc8kab4/checkpoint_epoch30+iter96875.pth
```
* Evaluate the model on the SCAN MCD1 splits by running:
```
export MODEL_CKPT=...
./eval_scripts/generate-order-deen.sh $MODEL_CKPT
```

## 4. Decode the undirected generation model with heuristic orders
* Left2Right
```
./eval_scripts/generate-deen.sh left_right_greedy_1iter
```

* Least2Most
```
./eval_scripts/generate-deen.sh least_most_greedy_1iter
```

* EasyFirst
```
./eval_scripts/generate-deen.sh easy_first_greedy_1iter
```

* Uniform
```
./eval_scripts/generate-deen.sh uniform_greedy_1iter
```

## Citation
```
@inproceedings{jiang-bansal-2021-learning-analyzing,
    title = "Learning and Analyzing Generation Order for Undirected Sequence Models",
    author = "Jiang, Yichen  and
      Bansal, Mohit",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.298",
    pages = "3513--3523",
}