export PYTHONPATH=$(pwd)
python train/train_generative.py\
    --data_dir=/Users/israfelsalazar/Documents/deep-genomic.nosync/deep-genomic/data/sim1\
    --condition_files var_current.csv\
    --c_norm\
    --num_classes=1\
    --z_dim=6\
    --c_embedded=64\
    --model=prior_vae\
    --is_conditional\
    --do_haploidization\
    --batch_size=75\
    --criterion=priorbce_elbo\
    --num_epochs=10\
    --do_eval\
    --do_encode\
    --compute_r2\
    --wandb_run_name=pcvae