# ------------------------------------
# WhobPyT Data Fetcher Functionalities
# ------------------------------------
#
# Modified directly from code in grifflab/kernel-flow-tools
# Inspired by nilearn & mne
#

# Importage
import os,sys,glob,shutil,numpy as np, pandas as pd
import requests, zipfile,gdown
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

WHOBPYT_DATA_FOLDER = '~/.whobpyt/data'


def get_localdefaultdatapath():
    return os.path.expanduser(WHOBPYT_DATA_FOLDER)


def pull_file(dlcode,destination,download_method):

  dest_file = destination.split('/')[-1]
  print('\n\nDownloading %s\n' %dest_file)

  if download_method == "wget": 
    os.system('wget %s -O %s' %(dlcode, destination))

  elif download_method == "gdown":
    url = "https://drive.google.com/uc?id=" + dlcode
    gdown.download(url, destination, quiet=False)

  elif download_method == "requests":
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(
    url, params={"id": dlcode}, stream=True)
    # get the confirmation token to download large files
    token = None
    for key, value in response.cookies.items():
      if key.startswith("download_warning"):
        token = value
    if token:
      params = {"id": id, "confirm": token}
      response = session.get(url, params=params, stream=True)
    # save content to the zip-file  
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
      for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:
          f.write(chunk)


def pull_folder(dlcode: str, newfolder_name: str, dest_folder = None):

    # Creates a new folder `newfolder_name` inside `dest_folder`, 
    # downloads the data into that location, 
    # and returns the location as a string

    # Assemble gdown gdrive url target
    url = "https://drive.google.com/drive/folders/" + dlcode
  
    # (dest_folder is the location this folder will be downloaded into)

    # if no dest folder supplied, use the default location in user's home    
    if dest_folder is None: dest_folder = get_localdefaultdatapath()
    newfolder = os.path.join(dest_folder, newfolder_name)

    # Create dest folder if it does not exist
    if not os.path.exists(newfolder): os.makedirs(newfolder) 

    # cd to dest folder, grab the download, cd back to current dir  
    cwd = os.getcwd()
    os.chdir(newfolder)
    gdown.download_folder(url, remaining_ok=True)
    os.chdir(cwd)

    print('\n\nDownloaded \n\n%s \n\nto \n\n%s\n\n' %(url,newfolder))           
    return newfolder


def fetch_hcpl2k8(dest_folder = None):

    dlcode = "18smy3ElTd4VksoL4Z15dhwT5l3yjk6xS"

    res_location = pull_folder(dlcode, dest_folder=dest_folder, 
                               newfolder_name = 'hcpl2k8')

    return res_location

   


def fetch_MomiEtAlELife2023(dest_folder=None):
    #
    # -----
    # Usage
    # -----
    #
    # thisdir = os.getcwd()
    # dest_folder = os.path.join(thisdir,'reproduce_Momi_et_al_2023')
    #
    # res = fetch_MomiEtAlElife2023(dest_folder=dest_folder)
    #

    cwd = os.getcwd()
   
    # Life hack to stop for erroring on the next line
    if dest_folder is None: dest_folder = ''

    # Create the target out dir
    if not os.path.isdir(dest_folder): os.makedirs(dest_folder)
    os.chdir(dest_folder)

    # Pull the data folder
    newf_name = ''
    dlcode = '1t-9m0E88xUUcGWWH024H32maQUs_Jjgx'  # This is the DATA_LITE folder
    # dlcode = '1iwsxrmu_rnDCvKNYDwTskkCNt709MPuF' This is the FULL data folder (too large to download)
    res_loc_1 = pull_folder(dlcode, dest_folder=dest_folder,newfolder_name=newf_name)

    # Pull the fsaverage folder
    newf_name = '' 
    dlcode = '1YPyf3h9YKnZi0zRwBwolROQuqDtxEfzF'
    res_loc_2 = pull_folder(dlcode, dest_folder=dest_folder,newfolder_name=newf_name)

    # Confirm you got everything and go back to initial dir
    os.chdir(cwd)




def fetch_egtmseeg(dest_folder=None, redownload=False):
    #
    # -----
    # Usage
    # -----
    #
    # res = fetch_egtmseeg()
    #

    osf_url_pfx = 'https://osf.io/download'
    files_dict = {'680fc6eb5210d93da17c5a1d':'Schaefer2018_200Parcels_7Networks_count.csv',
                  '680fe69f9ba48ef173568c07':'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.txt',
                  '680fc6fbab92f2b5627c5a0f':'stim_weights.npy',
                  '680fc6f63068d66fc8568e3e':'Subject_1_low_voltage.fif',
                  '680fc6f80959f56dde568f29':'Subject_1_low_voltage_fittingresults_stim_exp.pkl',
                  '680fc6f99f030183213cbfd3':'Subject_1_low_voltage_lf.npy'}
    
    cwd = os.getcwd()
   
    if dest_folder is None:
        defpath = get_localdefaultdatapath()
        dest_folder = os.path.join(defpath, 'eg__tmseeg')

    # If input instruction was to re-download and folder is already present, remove it
    if os.path.isdir(dest_folder) and redownload == True:
        os.system('rm -rf %s' %dest_folder)

    # If the folder does not exist, create it and download the files
    if not os.path.isdir(dest_folder): 

        os.makedirs(dest_folder)
    
        os.chdir(dest_folder)

        for file_code, file_name in files_dict.items():
          dlcode = osf_url_pfx + '/' + file_code
          pull_file(dlcode, file_name, download_method='wget')
   
        os.chdir(cwd)

    return dest_folder 

def fetch_egismail2025(dest_folder=None, redownload=False):
    """
    Fetch multiple files for Ismail2025 using pull_file function.
    """

    cwd = os.getcwd()

    if dest_folder is None:
        defpath = get_localdefaultdatapath()
        dest_folder = os.path.join(defpath, 'eg__ismail2025')

    # If input instruction was to re-download and folder is already present, remove it
    if os.path.isdir(dest_folder) and redownload == True:
        shutil.rmtree(dest_folder)

    # Create the destination folder if it doesn't exist
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    os.chdir(dest_folder)

    files_to_download = [
        ('1P4WSVLiWDdoK_S2cSlDyt1V7JuphfsaZ', 'distance.txt'),
        ('1hubOoaJcCExawBKk-fpxNXSmSi-dx2-v', 'emp_noise_source.npy'),
        ('1frpLiDxwdduE1LAOWGPCSDTETTWjkKUh', 'emp_verb_source.npy'),
        ('1HpMJzTzxNn-YItSo_GJvCpI3EEuvVWUP', 'info.pkl'),
        ('1gtVxgl1z6QQKDyuxlWdePUpU__i2Njba', 'leadfield_3d.mat'),
        ('1rgaPu3fRPYxRb6EHkFac4ahC0McbBBJb', 'noise_evoked.npy'),
        ('1Lr3VV69jBVkNWqz7iPxJHVmflgBTsyYq', 'sim_noise_sensor.npy'),
        ('18os1jbGZS2si7_0p1bb7nplHEKYqW54J', 'sim_noise_source.npy'),
        ('1pSVY_YPic5mFAoU3xX99HzejS5Fq0ezW', 'sim_verb_sensor.npy'),
        ('1DCLLXr7e6y4o9Gs9sQhGDWrRO6HJU7Oe', 'sim_verb_source.npy'),
        ('1VKCFgHQ78rraTvyJxVWBYevm4omMYMxG', 'verb_evoked.npy'),
        ('1zFBPr25WZEPJVICLx19-JS3XQESsQVNu', 'weights.csv'),
    ]

    total_files = len(files_to_download)

    for idx, (dlcode, output_filename) in enumerate(files_to_download, start=1):
        print(f"Downloading file {idx} of {total_files}: {output_filename}")
        destination_path = os.path.join(dest_folder, output_filename)
        pull_file(dlcode, destination_path, download_method="gdown")

    os.chdir(cwd)

    return dest_folder

def  fetch_egmomi2025(dest_folder=None, redownload=False):
    #
    # -----
    # Usage
    # -----
    #
    # res = fetch_egmomi2025()
    #
    api = KaggleApi()
    api.config_values['username'] = "claires03"
    api.config_values['key'] = "ee39084a8974336d9fff7e1ced807e64"
    files_dict = {'davi1990/empirical-data':'empirical-data',
                  'davi1990/anatomical':'anatomical',
                  'davi1990/calculate-distance':'calculate-distance',
                  'davi1990/virtual-dissection':'virtual-dissection'}
    cwd = os.getcwd()
    #
    if dest_folder is None:
        defpath = get_localdefaultdatapath()
        dest_folder = os.path.join(defpath, 'eg__momi2025')
    # If input instruction was to re-download and folder is already present, remove it
    if os.path.isdir(dest_folder) and redownload == True:
        os.system('rm -rf %s' %dest_folder)
    # If the folder does not exist, create it and download the files
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
        os.chdir(dest_folder)
        for file_code, file_name in files_dict.items():
          dest_sub_folder = os.path.join(dest_sub_folder, file_name)
          if not os.path.isdir(dest_sub_folder):
              os.makedirs(dest_sub_folder)
              api.dataset_download_files(file_code, path= dest_sub_folder, unzip=True)
    return dest_folder
