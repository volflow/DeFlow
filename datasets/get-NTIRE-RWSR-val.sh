mkdir NTIRE-RWSR;
cd NTIRE-RWSR;

wget https://data.vision.ee.ethz.ch/alugmayr/NTIRE2019/public/Corrupted-va-x.zip;
unzip Corrupted-va-x.zip -d Corrupted-va-x && rm Corrupted-va-x.zip

wget https://data.vision.ee.ethz.ch/alugmayr/real-world-sr/NTIRE20/track1-valid-gt.zip;
unzip track1-valid-gt.zip && rm track1-valid-gt.zip

echo "Preparing Dataset for DeFlow";
cd ..;
python create_DeFlow_train_dataset.py -source ./NTIRE-RWSR/Corrupted-va-x/ -target ./NTIRE-RWSR/Corrupted-va-x/ -scales 1 4;
rm ./NTIRE-RWSR/Corrupted-va-x/*.png;

python create_DeFlow_train_dataset.py -source ./NTIRE-RWSR/track1-valid-gt/ -target ./NTIRE-RWSR/track1-valid-gt/ -scales 1 4 16;
rm ./NTIRE-RWSR/track1-valid-gt/*.png;