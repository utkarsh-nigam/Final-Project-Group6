from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

my_files=os.listdir('by_class/')
ignore_files=[".DS_Store","error_data.csv","image_data.csv"]

new_wd=my_path+'/by_class/'
os.chdir(new_wd)
flag=0
e_flag=0
data_frame=pd.DataFrame()
error_frame=pd.DataFrame()

for fileName in my_files:
    if (fileName not in ignore_files):
        files_inside=os.listdir(fileName+'/')
        print(fileName)
        for file_in in files_inside:
            if "train" in file_in:
                images_list = os.listdir(fileName + '/'+file_in+'/')
                for image in images_list:
                    try:
                        image_in = Image.open(fileName + '/'+file_in+'/'+image).convert('L')#.convert('1')
                        image_in = image_in.resize((56, 56), Image.ANTIALIAS)
                        data = np.asarray(image_in)
                        data = data.ravel()
                        data = np.array([data.ravel()])
                        temp_df = pd.DataFrame(data=data)
                        temp_df["Filename"]=fileName
                        if (flag==0):
                            data_frame = temp_df.copy()
                            flag=1
                        else:
                            data_frame = data_frame.append(temp_df)
                    except Exception as e:
                        error_array=[fileName,file_in,image,e]
                        temp_df = pd.DataFrame(data=error_array)
                        if (e_flag==0):
                            error_frame=temp_df.copy()
                            e_flag=1
                        else:
                            error_frame = error_frame.append(temp_df)
            data_frame.to_csv("all_data_scanned.csv")
            error_frame.to_csv("error_data.csv")
