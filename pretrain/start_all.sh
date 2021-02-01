python train.py \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir ../coder_all \
    --device cuda:0 --lang all \
    --save_steps 100000 \
    --max_steps 1000000 \
    --trans_loss_type CosineDistMult_MS --use_re true --use_rel true
