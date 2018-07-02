python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 3 --threads 2 --max-steps 63000 --random-seed 0 --triplet-margin 2.0 --device /gpu:1 --postfix 107 --out-channels 128

exit
