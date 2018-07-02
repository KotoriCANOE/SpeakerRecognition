python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 4 --threads 2 --max-steps 127000 --random-seed 0 --triplet-margin 2.0 --device /gpu:1 --postfix 109 --out-channels 256 --pretrain-dir train105.tmp

exit

python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 3 --threads 2 --max-steps 63000 --random-seed 0 --triplet-margin 2.0 --device /gpu:1 --postfix 107 --out-channels 128
