
exit

python data_pack.py $HOME/Datasets/Speech/VoxCeleb2/vox2_test_wav $HOME/Datasets/Speech/VoxCeleb2/vox2_test_npz --num-epochs 1 --test
python data_pack.py $HOME/Datasets/Speech/VoxCeleb2/vox2_dev_wav $HOME/Datasets/Speech/VoxCeleb2/vox2_dev_npz --num-epochs 4
python data_pack.py $HOME/Datasets/Speech/Corpus $HOME/Datasets/Speech/Corpus-others/npz --num-epochs 16
