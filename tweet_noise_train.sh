#!/usr/bin/env
alias train_tweet='nohup python -m bin.train \
  --config_paths="
      ./tweet/configs/tweet_auto.yml,
      ./tweet/configs/train_seq2seq.yml,
      ./tweet/configs/text_metrics_slice.yml,
      ./tweet/configs/run_params.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --output_dir $MODEL_DIR 2>&1 1>train_model_2_128h_autonoise_2e_2d_drop0.5b.log &'
train_tweet
