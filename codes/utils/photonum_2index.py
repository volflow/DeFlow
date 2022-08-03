import os
all_photos=os.listdir('D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/canon/4x')
photo_nums=[]
for photo in (all_photos):
    photo_nums.append(int(photo.split('.')[0]))
print(photo_nums)
print(sorted(photo_nums))
for index,val in enumerate(sorted(photo_nums)):
    os.rename(f'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/canon/4x/{val}.jpg',f'D:/Deep_Project/DeFlow/datasets/DPED-RWSR/train/canon/4x/{index}.jpg')
    print(index,val)