export PYTHONPATH=$(pwd)
python train/train_generative.py\
    --data_dir=/Users/israfelsalazar/Documents/deep-genomic.nosync/deep-genomic/data/sim_1\
    --condition_files="var_current.csv"\
    --num_classes=1\
    --z_dim=6\
    --is_conditional\
    --model=simple_vae\
    --do_haploidization\
    --batch_size=8\
    --criterion="bce_elbo"\
