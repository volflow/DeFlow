mkdir NTIRE-RWSR;
cd NTIRE-RWSR;

wget https://data.vision.ee.ethz.ch/alugmayr/NTIRE2019/public/Corrupted-tr-x.zip;
unzip Corrupted-tr-x.zip -d Corrupted-tr-x && rm Corrupted-tr-x.zip

wget https://data.vision.ee.ethz.ch/alugmayr/NTIRE2019/public/Corrupted-tr-y.zip;
unzip Corrupted-tr-y.zip -d Corrupted-tr-y && rm Corrupted-tr-y.zip

echo "Preparing Dataset for DeFlow";
cd ..;
python create_DeFlow_train_dataset.py -source ./NTIRE-RWSR/Corrupted-tr-x/ -target ./NTIRE-RWSR/Corrupted-tr-x/ -scales 1 4;
rm ./NTIRE-RWSR/Corrupted-tr-x/*.png;

python create_DeFlow_train_dataset.py -source ./NTIRE-RWSR/Corrupted-tr-y/ -target ./NTIRE-RWSR/Corrupted-tr-y/ -scales 1 4 16;
rm ./NTIRE-RWSR/Corrupted-tr-y/*.png;
