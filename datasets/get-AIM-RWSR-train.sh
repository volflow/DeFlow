mkdir AIM-RWSR;
cd AIM-RWSR;
wget https://data.vision.ee.ethz.ch/alugmayr/real-world-sr/AIM19/train-noisy-images.zip;
unzip train-noisy-images.zip && rm train-noisy-images.zip

wget https://data.vision.ee.ethz.ch/alugmayr/real-world-sr/AIM19/train-clean-images.zip;
unzip train-clean-images.zip && rm train-clean-images.zip


echo "Preparing Dataset for DeFlow";
cd ..;
python create_DeFlow_train_dataset.py -source ./AIM-RWSR/train-noisy-images/ -target ./AIM-RWSR/train-noisy-images/ -scales 1 8;
rm ./AIM-RWSR/train-noisy-images/*.png;

python create_DeFlow_train_dataset.py -source ./AIM-RWSR/train-clean-images/ -target ./AIM-RWSR/train-clean-images/ -scales 1 4 32;
rm ./AIM-RWSR/train-clean-images/*.png;
