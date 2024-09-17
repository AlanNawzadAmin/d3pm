
python train.py \
    data.data=lm1b \
    +data.valid=lm1b \
    +data.tokenizer_name_or_path=bert-base-uncased \
    +data.cache_dir=/scratch/nvg7279/sched_diff/data_cache \
    +data.wrap=True \
    model.forward_kwargs.type=bert_embed \
    model.model=ScheduleCondition \
    architecture.x0_model_class=DIT \
    +architecture.nn_params.n_blocks=8 \
    +architecture.nn_params.hidden_size=512 \
    +architecture.nn_params.n_heads=8 \
    +architecture.nn_params.cond_dim=128 \
    +architecture.nn_params.dropout=0.1 \
    +architecture.nn_params.scale_by_sigma=True \
    +architecture.nn_params.tie_word_embeddings=False \
    +architecture.nn_params.length=1024 \
