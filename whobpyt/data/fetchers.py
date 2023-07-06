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

WHOBPYT_DATA_FOLDER = '~/.whobpyt/data'


def get_localdefaultdatapath():
    return os.path.expanduser(WHOBPYT_DATA_FOLDER)


def pull_file(dlcode,destination,download_method):

  dest_file = destination.split('/')[-1]
  print('\n\nDownloading %s\n' %dest_file)

  if download_method == "gdown":

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



