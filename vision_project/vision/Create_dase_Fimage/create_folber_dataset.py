from vision.Config import image_dir, image_val_dir,ann_file, ann_file_val
from pycocotools.coco import COCO
import json
import os
import requests
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import requests
from pycocotools.coco import COCO
import shutil



class getDataFoalberCoco:
  def __init__(self,
               image_dir: str ,
               ann_file_1 : str,
               list_catecoy : list,
               destination_dir : str
                ):
    self.image_dir = image_dir
    self.ann_file  = ann_file_1
    self.list_catecoy = list_catecoy
    self.destination_dir = destination_dir


  def get_image_label(self):
    '''
    get data and label in list
    self.list_data[{'image':img,'anns':anns}, ...]
    '''
    if  hasattr(self, 'list_data'):
      return None
    print('in function   :  get_image_lable')
    with open(self.ann_file, 'r') as f:
      data = json.load(f)
    coco = COCO(self.ann_file)
    self.list_data = []
    for i in range(len(data['images'])):
      # Find image ID based on file name
      image_id = data['images'][i]['id']
      '''
      for image_info in data['images']:
          if image_info['file_name'] == '000000397133.jpg':
              image_id = image_info['id']
              break
      '''
      if image_id is not None:
          # Load image data using the found image ID
          img = coco.loadImgs(image_id)[0]
          anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))
          #cats = coco.loadCats(coco.getCatIds())
          #labels = [cat['name'] for cat in cats]

          self.list_data.append({'image':img,'anns':anns})
      else:
          print(f"Image with number image  {i}  not found in the dataset.")




  def get_category_name(self):
    '''
     list of category_id
    -> self.caregory_list
    '''
    if   hasattr(self, 'category_list') :
      return None
    print('in function   :  get_category_name')
    self.get_image_label()
    self.category_list = []

    for i  in self.list_data:
      list_new = [i1['category_id']for i1 in i['anns']]
      self.category_list.append(list_new)





  def copy_image(self, data, categore_name : str):
      """Copies images to a new directory based on the provided list_data.
      Args:
          list_data: The list of image data from the COCO dataset.
          destination_dir: The directory to copy the images to.
      """
      destination_dir1 = os.path.join(self.destination_dir, categore_name)
      if not os.path.exists(destination_dir1):
          os.makedirs(destination_dir1)



      image_info = data['image']
      image_path = os.path.join(self.image_dir, image_info['file_name'])
      destination_path = os.path.join(destination_dir1, image_info['file_name'])
      # Checks if data exists and has not been copied
      if (not os.path.exists(destination_path)) and os.path.exists(image_path):
          shutil.copy2(image_path, destination_path) # copy2 preserves metadata
          #print(f"Copied {image_info['file_name']} to {destination_path}")
      elif not os.path.exists(image_path):
          print(f"Warning: Image file not found: {image_path}")


  def get_new_category_list(self):
    '''
    -> self.new_category_list
    '''
    if hasattr(self, 'new_category_list'):
      return None
    print('in function   :  get_new_category_list')
    self.get_category_name()
    self.new_category_list = []
    for i in self.category_list:

      a = np.array(i)
      b = np.array(self.list_catecoy)

      label = np.intersect1d(a, b)
      label = label if label.size > 0 else [-1]
      self.new_category_list.append(label)


  def create_dataset_folber(self):
    '''
    create and save dataset image folber
    '''
    self.get_new_category_list()
    len_all_data  = len(self.new_category_list)
    for i in tqdm(range(len_all_data)):
      for i1 in self.new_category_list[i]:
        self.copy_image(data = self.list_data[i], categore_name  =  str(i1))




if __name__ == "__main__":


  '''

  train_data = getDataFoalberCoco(
               image_dir  ,
               ann_file,
               [1],
               '/content/data_set/train_dataset')

  train_data.create_dataset_folber()

  
  val_data =  getDataFoalberCoco(
               image_val_dir  ,
               ann_file_val,
               [1],
               '/content/data_set/val_dataset')

  val_data.create_dataset_folber()'''