python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 4 --threads 2 --max-steps 63000 --random-seed 0 --triplet-margin 2.0 --device /gpu:0 --postfix 110 --out-channels 64

exit

python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 4 --threads 2 --max-steps 63000 --random-seed 0 --triplet-margin 2.0 --device /gpu:0 --postfix 105
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 3 --threads 2 --max-steps 63000 --random-seed 0 --triplet-margin 2.0 --device /gpu:0 --postfix 106 --out-channels 64
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 4 --threads 2 --max-steps 127000 --random-seed 0 --triplet-margin 2.0 --device /gpu:0 --postfix 108 --out-channels 64 --pretrain-dir train106.tmp
