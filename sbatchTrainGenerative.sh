export PYTHONPATH=$(pwd)
python train/train_generative.py\
    --data_dir=/Users/israfelsalazar/Documents/deep-genomic.nosync/deep-genomic/data/sim1\
    --condition_files var_current.csv pop.csv\--condition_files var_current.csv pop.csv\
    --num_classes=1\
    --z_dim=6\
    --model=simple_vae\
    --is_conditional\
    --do_haploidization\
    --batch_size=75\
    --criterion=bce_elbo\
    --num_epochs=200\
    --do_eval\
    --do_encode\
    --compute_r2\
    --wandb_run_name=simple_cvae\