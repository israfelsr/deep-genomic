export PYTHONPATH=$(pwd)
python train/train_generative.py\
    --data_dir=/Users/israfelsalazar/Documents/deep-genomic.nosync/deep-genomic/data/sim1\
    --output_dir=/Users/israfelsalazar/Documents/deep-genomic.nosync/deep-genomic/experiment_outputs\
    --condition_files var_current.csv\
    --c_norm\
    --z_dim=2\
    --model=simple_vae\
    --do_haploidization\
    --batch_size=75\
    --criterion=bce_elbo\
    --num_epochs=100\
    --do_eval\
    --wandb_run_name=vanilla_vae