برای کمک به استفاده آسان‌تر از کدهایی که در فایل‌هایت بارگذاری کرده‌ای، در ادامه یک **راهنمای کامل استفاده (Usage Guide)** برای توابع مهم در هر دو فایل `im_show.py` و `train_val_functiones.py` ارائه می‌دهم:

---

## 📁 فایل `im_show.py` — توابع نمایش و مدیریت تصاویر

---

### 1. `imshow_function(...)`

**نمایش چند تصویر به همراه عنوان رنگی در یک figure**

```python
imshow_function(
    img_list=[...],             # لیست تصاویر (Tensorهای PyTorch، مثل [C, H, W])
    color_list=['red', ...],    # رنگ عنوان هر تصویر
    title_list=['label1', ...], # لیست عنوان برای هر تصویر
    row=None,                   # تعداد ردیف‌ها (اختیاری)
    colum=3                     # تعداد ستون‌ها (پیش‌فرض: 3)
)
```

---

### 2. `plot_random_samples(model, test_dataset, num_samples=5, show_images='all F')`

**انتخاب تصادفی نمونه‌ها از دیتاست تست و نمایش پیش‌بینی مدل روی آن‌ها**

```python
plot_random_samples(
    model=my_model,
    test_dataset=test_ds,
    num_samples=9,
    show_images='all'  # یا 'TT', 'TF', 'FT', 'FF', 'all T', 'all F'
)
```

---

### 3. `tree_directory_images(directory_path)`

**نمایش ساختار درختی پوشه تصاویر و نمایش تعداد و نوع فایل‌ها**

```python
tree_directory_images('/path/to/dataset')
```

---

### 4. `move_or_copy_images_to_folder(...)`

**انتقال یا کپی تصاویر از یک پوشه به پوشه دیگر با قابلیت افزودن پسوند**

```python
move_or_copy_images_to_folder(
    source_folder='images/source',
    destination_folder='images/destination',
    pasvand='cat',                        # مثل: cat_001.jpg
    extensions=['.jpg', '.png'],
    move_or_copy='move'                  # یا 'copy'
)
```

---

## 📁 فایل `train_val_functiones.py` — آموزش، ارزیابی و ذخیره‌سازی مدل

---

### 1. `train_step(...)`

**انجام یک مرحله آموزشی همراه با ارزیابی دوره‌ای و ذخیره نتایج**

```python
results = train_step(
    model=model,
    dataloader=train_loader,
    loss_fn=loss_function,
    optimizer=optimizer,
    dataloader_test=test_loader,
    number_ep=100                      # هر 100 batch یکبار نمایش نتایج
)
```

---

### 2. `test_step(...)`

**ارزیابی مدل روی داده تست و محاسبه دقت و خطا**

```python
test_loss, test_acc = test_step(
    model=model,
    dataloader=test_loader,
    loss_fn=loss_function,
    show_plot_and_F1=True             # رسم confusion matrix
)
```

---

### 3. `train(...)`

**اجرای کل مراحل آموزش به‌صورت multi-epoch**

```python
results = train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    loss_fn=loss_function,
    epochs=10,
    number_ep=100
)
```

---

### 4. `save_models_historyes(...)`

**ذخیره مدل و تاریخچه آموزش به صورت `.pth` و `.csv`**

```python
save_models_historyes(
    model=model,
    history=results,
    dir_save='checkpoints',
    name_file='model_v1'
)
```

---

### 5. `load_model(...)`

**بارگذاری مدل ذخیره‌شده از فایل**

```python
model = load_model(
    model_class=MyModel,                         # کلاس مدل
    checkpoint_path='checkpoints/model_v1.pth',  # مسیر فایل
    device='cuda'                                # یا 'cpu'
)
```

---

