from vision.Config import image_dir, image_val_dir,ann_file, ann_file_val
from pycocotools.coco import COCO
if __name__ == "__main__":
  # بارگذاری فایل آنوتیشن COCO (آدرس فایل خود را قرار دهید)
  coco = COCO(ann_file_val)  # آدرس فایل آنوتیشن خود را قرار دهید

  # استخراج لیست کلاس‌ها (دسته‌بندی‌ها)
  categories = coco.loadCats(coco.getCatIds())


  print(categories)
