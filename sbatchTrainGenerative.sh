export PYTHONPATH=$(pwd)
python train/train_generative.py\
    --data_dir=/content/deep-genomic/data/news/hchp\
    --output_dir=/content/deep-genomic/resuls/experiment_hchp\
    --condition_files var_current.csv\
    --is_conditional\
    --do_eval\
    --compute_r2\
    --use_wandb\
    --c_norm\
    --do_haploidization\
    --z_dim=6\
    --num_classes=1\
    --model=prior_vae\
    --batch_size=120\
    --criterion=elbo_prior\
    --num_epochs=100\
    --wandb_run_name=test_pandas