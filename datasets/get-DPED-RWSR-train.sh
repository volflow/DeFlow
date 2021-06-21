mkdir DPED-RWSR;
cd DPED-RWSR;
wget https://data.vision.ee.ethz.ch/alugmayr/NTIRE2019/public/DPEDiphone-tr-x.zip;
unzip DPEDiphone-tr-x.zip -d DPEDiphone-tr-x && rm DPEDiphone-tr-x.zip

## DeFlow does not use the given high quality images 
## as we found that the scenes are too different
## instead we treat 4x downsampled low-quality images as the high quality images 
# wget https://data.vision.ee.ethz.ch/alugmayr/NTIRE2019/public/DPEDiphone-tr-y.zip;
# unzip DPEDiphone-tr-y.zip -d DPEDiphone-tr-y && rm DPEDiphone-tr-y.zip

echo "Preparing Dataset for DeFlow";
cd ..;
python create_DeFlow_train_dataset.py -source ./DPED-RWSR/DPEDiphone-tr-x/ -target ./DPED-RWSR/DPEDiphone-tr-x/ -scales 1 4 16;
rm ./DPED-RWSR/DPEDiphone-tr-x/*.png;
