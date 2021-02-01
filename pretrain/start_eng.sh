python train.py \
    --output_dir ../coder_eng \
    --device cuda:0 --lang eng \
    --save_steps 100000 \
    --max_steps 100000 \
    --trans_loss_type CosineDistMult_MS --use_re true --use_rel true
