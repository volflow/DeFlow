mkdir DPED-RWSR;
cd DPED-RWSR;
wget https://data.vision.ee.ethz.ch/alugmayr/NTIRE2019/public/DPEDiphone-va.zip;
unzip DPEDiphone-va.zip -d DPEDiphone-va && rm DPEDiphone-va.zip

echo "Preparing Dataset for DeFlow";
cd ..;
python create_DeFlow_train_dataset.py -source ./DPED-RWSR/DPEDiphone-va/ -target ./DPED-RWSR/DPEDiphone-va/ -scales 1 4;
rm ./DPED-RWSR/DPEDiphone-va/*.png;
