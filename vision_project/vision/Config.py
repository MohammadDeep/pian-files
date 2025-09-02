from pathlib import Path
# dirctory dataset 
base_dir = Path('./__dataset__coco') # dase directory
image_dir = Path(base_dir, 'train2017')  # train orginal image direcory
ann_file = Path(base_dir, 'annotations/instances_train2017.json')  # label train data dir
image_val_dir = Path(base_dir, 'val2017/') # val orginal image direcory
ann_file_val = Path(base_dir,'annotations/instances_val2017.json') # label val data dir

# list of data classes for 
list_calsses = [1]

# dirctory for save dataset foder for classes
dir_dataset_folder = Path(base_dir, 'data_set_floders')
dir_dataset_orgnal = Path(dir_dataset_folder , 'dataset_orgmal')
dir_image_gent = Path(dir_dataset_folder, 'CutImage')
dir_dataset_folder_val = Path(base_dir, 'data_set_floders_val')
dir_dataset_orgnal_val = Path(dir_dataset_folder_val , 'dataset_orgmal')
dir_image_gent_val = Path(dir_dataset_folder_val, 'CutImage')
# folber names
''' 
save image in:
dir_dataset_folber|---> folber_name_bake_image_Not
                  |--->folber_name_box_image
                  |---> flober_name_cut_image

'''
folber_name_bake_image_Not = 'bake_image_Not'
folber_name_box_image = 'box_image'
folber_name_cut_image = 'cut_image'

dir_bake_image_Not_val = Path(dir_image_gent_val , folber_name_bake_image_Not)
dir_box_image_val = Path(dir_image_gent_val , folber_name_box_image)
dir_cut_image_val = Path(dir_image_gent_val , folber_name_cut_image)
dir_core_class_val = Path(dir_dataset_orgnal_val, str(list_calsses[0]))
dir_non_core_class_val = Path(dir_dataset_orgnal_val, '-1')

dic_dir_val = {
    'core_class' : dir_core_class_val,
    'non_core_class' : dir_non_core_class_val,
    'bake_image_Not' : dir_bake_image_Not_val,
    'box_image' : dir_box_image_val,
    'cut_image' : dir_cut_image_val
    }
# فولدر های که برای  درست کردن دیتا ست به پوشه های کلاس یک و کلاس دو انتفال داده می شود
dic_name_folber_clases = {
    
    f'class_id{list_calsses[0]}':['core_class' ,'cut_image' , 'box_image'],
    'calss_other':['non_core_class', 'bake_image_Not']

    }
dir_bake_image_Not = Path(dir_image_gent , folber_name_bake_image_Not)
dir_box_image = Path(dir_image_gent , folber_name_box_image)
dir_cut_image = Path(dir_image_gent , folber_name_cut_image)
dir_core_class = Path(dir_dataset_orgnal, str(list_calsses[0]))
dir_non_core_class = Path(dir_dataset_orgnal, '-1')

dic_dir = {
    'core_class' : dir_core_class,
    'non_core_class' : dir_non_core_class,
    'bake_image_Not' : dir_bake_image_Not,
    'box_image' : dir_box_image,
    'cut_image' : dir_cut_image
    }

dic_name_folber_clases_val = {
    
    f'class_id{list_calsses[0]}':['core_class' ,'cut_image' , 'box_image'],
    'calss_other':['non_core_class', 'bake_image_Not']

    }
dic_name_cut_image_non_core_class = {
    
    f'class_id{list_calsses[0]}':['cut_image'],
    'calss_other':['non_core_class']

    }
dic_name_all_folber = {
    
    f'class_id{list_calsses[0]}':['core_class' ,'cut_image' , 'box_image'],
    'calss_other':['non_core_class', 'bake_image_Not']

    }


dir_extended_all_folber_val = Path(dir_dataset_folder_val , 'all_folber_dataset')
dir_extended_cut_image_non_core_class_val = Path(dir_dataset_folder_val , 'cut_image_non_core_class')


dir_extended_all_folber = Path(dir_dataset_folder , 'all_folber_dataset')
dir_extended_cut_image_non_core_class = Path(dir_dataset_folder , 'cut_image_non_core_class')


dic_dataset = {
    'all_folber_val' : {
        'dir' : dir_extended_all_folber_val,
        'dic_name': dic_name_all_folber,
        'dic_dir': dic_dir_val
    },
    'cut_image_non_core_class_val':{
        'dir' : dir_extended_cut_image_non_core_class_val,
        'dic_name': dic_name_cut_image_non_core_class,
        'dic_dir': dic_dir_val
    },
    'all_folber' : {
        'dir' : dir_extended_all_folber,
        'dic_name': dic_name_all_folber,
        'dic_dir': dic_dir
    },
    'cut_image_non_core_class':{
        'dir' : dir_extended_cut_image_non_core_class,
        'dic_name': dic_name_cut_image_non_core_class,
        'dic_dir': dic_dir
    } 

}

# urn code for Create data
Create_train_data = True
Create_val_data = True



# dir history and models save in
dir_history_model = Path('./__history_and_modeles/')
dir_history_model_google_dirve = Path('/content/drive/MyDrive')