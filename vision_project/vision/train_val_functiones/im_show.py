import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import models
import random
from tqdm import tqdm
import time
import os
import shutil
from typing import Literal
from pathlib import Path

def imshow_function(
      
      img_list, color_list,
        row=None, 
        colum=3, 
        title_list=None,
        titel = None,
        dir_save = None):
    """
    نمایش چند تصویر در یک شکل (figure) با استفاده از تنسورهای PyTorch.
    پارامترها:
      - img_list    : لیستی از تنسورهای تصاویر (هر تنسور شکل [C, H, W]).
      - color_list  : لیستی از رنگ‌ها برای عناوین (آبی برای درست، قرمز برای اشتباه، و غیره).
      - title_list  : لیستی از عناوین (متن) برای هر تصویر.
      - row         : تعداد ردیف‌های شبکه‌ی زیرمحورها. اگر None باشد، براساس تعداد تصاویر و ستون‌ها محاسبه می‌شود.
      - colum       : تعداد ستون‌های شبکه‌ی زیرمحورها در یک ردیف (پیش‌فرض 3).
    """
    num_imgs = len(img_list)
    if title_list is None:
        title_list = ["" for _ in range(num_imgs)]
    if color_list is None:
        color_list = ["black" for _ in range(num_imgs)]

    # اگر تعداد ردیف به صورت دستی مشخص نشده باشد،
    # تعداد ردیف = ceil(تعداد تصاویر / تعداد ستون)
    if row is None:
        row = (num_imgs + colum - 1) // colum

    fig, axes = plt.subplots(row, colum, figsize=(4*colum, 4*row))
    axes = axes.flatten()
    print(f'create plot...')
    for i in tqdm(range(row * colum)):
        ax = axes[i]
        if i < num_imgs:
            img_tensor = img_list[i]
            title = title_list[i]
            color = color_list[i]

            # تبدیل تنسور به numpy و تغییر ترتیب محورها از [C, H, W] به [H, W, C]
            img = img_tensor.cpu().numpy().transpose((1, 2, 0))

            # بازگرداندن نرمالیزه به حالت اولیه (در صورت استفاده از استاندارد ImageNet)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img  = std * img + mean
            img  = np.clip(img, 0, 1)

            ax.imshow(img)
            # اندازه تصویر (بعد از تبدیل تنسور)
            #height = img.shape[0]  # معادل H در [H, W, C]

            # محاسبه اندازه فونت بر اساس ارتفاع تصویر
            # به طور تجربی: font_size = ارتفاع تصویر / یک مقدار ثابت
            #font_size = min(12, height // 25)  # محدودیت حداقل
            # اندازه فیزیکی ساب‌پلات
            subplot_height_inches = fig.get_size_inches()[1] / row
            subplot_width_inches  = fig.get_size_inches()[0] / colum

            subplot_min = min(subplot_height_inches, subplot_width_inches)
            a = subplot_min / len(title)
            font_size = int(a / .55 * 72) + 1
            ax.set_title(title, color=color,fontsize=font_size)
            ax.axis('off')
        else:
            # اگر تصویری وجود ندارد (تعداد تصاویر کمتر از row*colum)، ساب‌پلات را خاموش می‌کنیم
            ax.axis('off')
    if titel:
        plt.suptitle(titel, fontsize=18)
    if dir_save:
       dir_save = Path(dir_save , f'{titel}.png' )
       plt.savefig(dir_save)

    plt.tight_layout()
    plt.show()


def plot_random_samples(model,
                        test_dataset,
                        num_samples=5,
                        figsize=(4, 4)
                        , show_images = 'all F' # 'all' or 'TT' or 'FF' or 'FT' or 'TF' or 'all T' or 'all F'
                         , titel = None,
                         dir_save = None,
                         threshold = 0.5,
                         use_sigmoid = True):
    """
    انتخاب نمونه‌های تصادفی از دیتاست تست، پیش‌بینی با مدل، و نمایش آنها.
    اگر پیش‌بینی درست باشد، عنوان با رنگ آبی و در غیر این صورت با رنگ قرمز نمایش داده می‌شود.
    """
    # دستگاه (CPU یا GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # برعکس کردن دیکشنری class_to_idx برای نمایش نام کلاس‌ها
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    print(f'idx_to_class: {idx_to_class}')

    total_images = len(test_dataset)
    if num_samples > total_images:
        print(f"تعداد درخواست‌شده ({num_samples}) بیشتر از تعداد کل تصاویر ({total_images}) است. تنظیم به {total_images}.")
        num_samples = total_images

    # انتخاب تصادفی اندیس‌ها
    random_indices = random.sample(range(total_images), num_samples)

    list_images = []
    list_titles = []
    list_colors = []
    index_TT =[]
    index_TF = []
    index_FF = []
    index_FT = []

    print('model in image...')
    ii = 0
    for idx in tqdm(random_indices):
        # بارگذاری تصویر و برچسب واقعی
        img_path, _ = test_dataset.imgs[idx]
        img_tensor, true_idx = test_dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)  # اضافه کردن بعد batch
        true_idx = torch.tensor([true_idx]).to(device)

        # پیش‌بینی مدل
        with torch.no_grad():
            outputs = model(img_input)
            # add sigmoid function 
            if use_sigmoid : 
             #print('use sigmoid function on output model')
             outputs = torch.sigmoid(outputs)
            # اگر بیش از دو کلاس داریم از argmax استفاده می‌کنیم
            if len(idx_to_class) > 2:
                _, pred_idx = torch.max(outputs, 1)
                pred_idx = pred_idx.item()
            else:
                # فرض بر این است که خروجی سینگولار است (مثلاً sigmoid)
                prob =  outputs[0][0]
                if prob > threshold:
                    pred_idx = 1
                else:
                    pred_idx = 0

        true_label = idx_to_class[int(true_idx.item())]
        pred_label = idx_to_class[int(pred_idx)]
        title_str = f"True: {true_label} | Pred: {pred_label} | output : {prob:1.2}"

        # تعیین رنگ عنوان
        if true_label == pred_label:
            color = 'blue'
            if true_label == idx_to_class[0]:
              index_FT.append(ii)
            else:
              index_TT.append(ii)

        else:
            color = 'red'
            if true_label == idx_to_class[0]:
              index_FF.append(ii)
            else:
              index_TF.append(ii)
        ii = ii + 1
        list_images.append(img_tensor)
        list_titles.append(title_str)
        list_colors.append(color)

    print(f'accuracy : {(len(index_TT) + len(index_FT))/ (len(list_colors))}')
    print(f'len(list_colors_True) : {len(index_TT) + len(index_FT)}')
    print(f'len(list_colors_False) : {len(index_TF) + len(index_FF)}')

    if show_images == 'TT' or show_images == 'all T' or show_images == 'all' :
      if len(index_TT) != 0:
        print('Plot TT')
        imshow_function(
            img_list=[list_images[i] for i in index_TT],
            title_list=[list_titles[i] for i in index_TT],
            color_list=[list_colors[i] for i in index_TT],
            colum=3,
            titel = f'{titel}__Plot TT' ,
            dir_save = dir_save
        )
      else:
         print('Number TT sample  = 0')
    if show_images == 'TF' or show_images == 'all F'or show_images == 'all' :
      if len(index_TF) != 0:
        print('Plot TF')
        imshow_function(
            img_list=[list_images[i] for i in index_TF],
            title_list=[list_titles[i] for i in index_TF],
            color_list=[list_colors[i] for i in index_TF],
            colum=3,
            titel = f'{titel}__Plot TF' ,
            dir_save = dir_save
        )
      else :
         print('Number TF sample  = 0')
    if show_images == 'FT' or show_images == 'all T'or show_images == 'all' :
      if len(index_FT) != 0:
        print('Plot FT')
        imshow_function(
            img_list=[list_images[i] for i in index_FT],
            title_list=[list_titles[i] for i in index_FT],
            color_list=[list_colors[i] for i in index_FT],
            colum=3,
            titel =f'{titel}__Plot FT' ,
            dir_save = dir_save
        )
      else:
        print('Number FT sample  = 0') 
    if show_images == 'FF'  or show_images == 'all F'or show_images == 'all':
      if len(index_FF) != 0:
        print('Plot FF')
        imshow_function(
            img_list=[list_images[i] for i in index_FF],
            title_list=[list_titles[i] for i in index_FF],
            color_list=[list_colors[i] for i in index_FF],
            colum=3,
            titel = f'{titel}__Plot FF' ,
            dir_save = dir_save
        )
    else:
        print('Number FF sample  = 0') 

# show tree file 
# prompt: نمودار درختی عکس های درون پوشه های تعداد عکس ها
# /content/train_dataset
# نوع عکس ها را هم مشخس کن
import os
def tree_directory_images(directory_path):
  """
  Creates a tree structure of image files within subdirectories,
  counting images and specifying their types.

  Args:
    directory_path: The path to the main directory.
  Example:
    tree_directory_iamges(directory_path  =  '/dataset')
  """
  print(f"{directory_path}")
  for root, dirs, files in os.walk(directory_path):
    # Get the level of the current directory relative to the starting directory
    level = root.replace(directory_path, '').count(os.sep)
    # Indent the output based on the directory level
    indent = '  ' * level
    # Get the base name of the current directory
    subdir = os.path.basename(root)
    if subdir: # Avoid printing the base directory name twice
        print(f'{indent}├── {subdir}/')

    # Count and list image files in the current directory
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if image_files:
        print(f'{indent}  ├── ({len(image_files)} images)')
        # Optional: Print the types of images
        image_types = set(os.path.splitext(f)[1].lower() for f in image_files)
        print(f'{indent}  │   ├── Types: {", ".join(image_types)}')

# Specify the path to the directory
directory_to_analyze = '/content/train_dataset'


# prompt: یک تابع که عکس ها موجود در یک پوشه را به پوشه دیگر انتقال دهد

def move_or_copy_images_to_folder(source_folder, destination_folder,
                          pasvand = None,
                          extensions=['.jpg', '.jpeg', '.png', '.gif'] ,
                          move_or_copy :Literal['move', 'copy'] = 'move'):
  """
  این تابع فایل‌های تصویری را از یک پوشه به پوشه دیگر انتقال می‌دهد.

  Args:
    source_folder: مسیر پوشه مبدأ.
    destination_folder: مسیر پوشه مقصد.
    extensions: لیستی از پسوندهای فایل تصویری برای انتقال.
    move_or_copy : ورودی های باید 'move'or 'copy'باشد

    # مثال استفاده:
    # source_folder = '/content/my_images'  # مسیر پوشه مبدأ خود را اینجا وارد کنید
    # destination_folder = '/content/moved_images' # مسیر پوشه مقصد خود را اینجا وارد کنید
    # move_images_to_folder(source_folder, destination_folder)

      
    """
  if move_or_copy not in ['move', 'copy']:
    move_or_copy = input('enter move_or_copy file (move , copy):')
  # اطمینان حاصل کنید که پوشه مقصد وجود دارد، در غیر این صورت آن را ایجاد کنید.
  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  # پیمایش در فایل‌های پوشه مبدأ
  for filename in tqdm(os.listdir(str(source_folder))):
    source_path = os.path.join(source_folder, filename)
    if  pasvand in filename:
      new_filename = filename
    else:
       new_filename = f'{pasvand}_{filename}'
    destination_path = os.path.join(destination_folder, new_filename )

    # بررسی کنید که آیا فایل یک فایل است و پسوند آن در لیست پسوندهای تصویر است
    if os.path.isfile(source_path) and any(filename.lower().endswith(ext) for ext in extensions):
      try:
        if move_or_copy == 'move':
            shutil.move(source_path, destination_path)
        elif move_or_copy == 'copy':
            shutil.copy(source_path, destination_path)
        #print(f"انتقال فایل: {filename}")
      except Exception as e:
        print(f"Error for move file : {filename} : {e}")




