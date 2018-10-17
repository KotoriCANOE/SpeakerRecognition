python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --num-labels 5994 --batch-size 72 --processes 2 --max-steps 511000 --random-seed 0 --device /gpu:1 --postfix 156 --embed-size 256 --normalization None

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
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 511000 --random-seed 0 --device /gpu:1 --postfix 130 --restore
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 132 --embed-size 64
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav" --processes 2 --threads 2 --max-steps 511000 --random-seed 0 --device /gpu:1 --postfix 132 --embed-size 64 --restore

python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 511000 --random-seed 0 --device /gpu:1 --postfix 134
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 511000 --random-seed 0 --device /gpu:1 --postfix 138

python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 139 --center-decay 0.5
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 140 --center-decay 0.9
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 141 --center-decay 0.95
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 2047000 --random-seed 0 --device /gpu:1 --postfix 141 --center-decay 0.95 --restore
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 127000 --random-seed 0 --device /gpu:1 --postfix 142 --center-decay 0.99
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz2" --packed --out-channels 5994 --processes 2 --max-steps 2047000 --random-seed 0 --device /gpu:1 --postfix 145 --generator-lr 1.4e-3

python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 2047000 --random-seed 0 --device /gpu:1 --postfix 146 --center-decay 0.95 --generator-lr 1.4e-3
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 2047000 --random-seed 0 --device /gpu:1 --postfix 148
python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --out-channels 5994 --processes 2 --max-steps 2047000 --random-seed 0 --device /gpu:1 --postfix 149 --embed-size 512

python train.py "$HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz" --packed --num-labels 5994 --batch-size 72 --processes 2 --max-steps 1023000 --random-seed 0 --device /gpu:1 --postfix 152 --embed-size 512 --normalization None
