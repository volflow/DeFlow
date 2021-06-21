mkdir AIM-RWSR;
cd AIM-RWSR;

echo "Downloading Datasets";
wget https://data.vision.ee.ethz.ch/alugmayr/real-world-sr/AIM19/valid-input-noisy.zip;
unzip valid-input-noisy.zip && rm valid-input-noisy.zip;

wget https://data.vision.ee.ethz.ch/alugmayr/real-world-sr/AIM19/valid-gt-clean.zip;
unzip valid-gt-clean.zip && rm valid-gt-clean.zip;

echo "Preparing Dataset for DeFlow";
cd ..;
python create_DeFlow_train_dataset.py -source ./AIM-RWSR/valid-input-noisy/ -target ./AIM-RWSR/valid-input-noisy/ -scales 1 8;
rm ./AIM-RWSR/valid-input-noisy/*.png;

python create_DeFlow_train_dataset.py -source ./AIM-RWSR/valid-gt-clean/ -target ./AIM-RWSR/valid-gt-clean/ -scales 1 4 32;
rm ./AIM-RWSR/valid-gt-clean/*.png;
