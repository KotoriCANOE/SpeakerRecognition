python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 511000 --random-seed 0 --device /gpu:1 --postfix 130 --restore

exit

python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 3 --threads 2 --max-steps 63000 --random-seed 0 --triplet-margin 2.0 --device /gpu:1 --postfix 107 --out-channels 128
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 4 --threads 2 --max-steps 127000 --random-seed 0 --triplet-margin 2.0 --device /gpu:1 --postfix 109 --out-channels 256 --pretrain-dir train105.tmp
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 4 --threads 2 --max-steps 63000 --random-seed 0 --triplet-margin 2.0 --device /gpu:1 --postfix 111 --out-channels 256 --batch-norm 0.999
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 4 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 113
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 1 --threads 4 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 115
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 116
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 118 --dropout 0.9
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 120
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 122 --triplet-margin 0.5
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 511000 --random-seed 0 --device /gpu:1 --postfix 122 --triplet-margin 0.5 --restore
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 124 --triplet-margin 0.5
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 128 --triplet-margin 0
