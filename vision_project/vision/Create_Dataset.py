from vision.Config import dir_extended_all_folber ,dic_dir , dic_name_folber_clases 
from vision.train_val_functiones.im_show import move_or_copy_images_to_folder, tree_directory_images
from pathlib import Path
def Create_dataset (dic_name = dic_name_folber_clases ,
                    dic_dir = dic_dir,
                    dir_extended = dir_extended_all_folber,
                    move_or_copy_files = 'move'
                    ):
    for class_name in dic_name:
        for folber_name in dic_name[class_name]:
            print(f'{move_or_copy_files} : {dic_dir[folber_name]} -> {Path(dir_extended, class_name)}')
            move_or_copy_images_to_folder(
                source_folder = dic_dir[folber_name],
                destination_folder = Path(dir_extended, class_name),
                pasvand = folber_name,
                move_or_copy= move_or_copy_files
            )

    tree_directory_images(str(dir_extended))

from vision.Config import dic_dataset

def Create_dataet_dir(name , dic_dataset = dic_dataset, move_or_copy_files = 'copy'):
    data_dir = dic_dataset[name]
    Create_dataset (dic_name = data_dir['dic_name'] ,
                    dic_dir = data_dir['dic_dir'],
                    dir_extended = data_dir['dir'],
                    move_or_copy_files = move_or_copy_files
                    )


