from pathlib import Path
from collections import OrderedDict
from typing import Optional, List, Tuple
import psutil
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import random
import matplotlib as plt



class CutImage:
    """
    Efficiently cut and composite objects from images based on COCO annotations,
    using a limited-size LRU cache with memory-aware eviction.
    """
    def __init__(
        self,
        image_dir: str,
        bake_dir: str,
        ann_file: str,
        category_id: int,
        destination_dir: str,
        cache_size: int = 1000,
        memory_threshold: float = 95.0 # [0, 100]
    ):
        # Initialize paths
        self.image_dir = Path(image_dir)
        self.bake_dir = Path(bake_dir)
        self.ann_file = Path(ann_file)
        self.destination_dir = Path(destination_dir)
        self.category_id = category_id
        self.bboxes_image :dict[str , List[List[float]]] = {}
        # LRU cache for loaded images
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._use_ram = True
        # Threshold for system memory usage (%) to trigger eviction
        self._memory_threshold = memory_threshold

        # Preload bake image paths for random selection
        self._bake_paths: List[Path] = (
            list(self.bake_dir.glob("*.jpg")) +
            list(self.bake_dir.glob("*.png")) +
            list(self.bake_dir.glob("*.jpeg"))
        )
        print(f'len image for bakcgund {len(self._bake_paths)}')

        # Initialize COCO API
        self._coco = COCO(str(self.ann_file))

    def _read_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Read image from disk with LRU caching. Evict oldest entries
        when cache is full or system memory usage exceeds threshold.
        """
        # Cache hit: move to end (most recently used)
        if path in self._cache:
            img = self._cache.pop(path)
            self._cache[path] = img
            return img

        # Check system memory and cache size for eviction
        mem_percent = psutil.virtual_memory().percent
        # Evict if memory usage too high
        while self._cache and mem_percent >= self._memory_threshold:
            evicted_path, _ = self._cache.popitem(last=False)
            if self._use_ram :
              self._use_ram  = False

              print(f"\n High memory usage {mem_percent:.1f}%  , evicted {evicted_path}")
            mem_percent = psutil.virtual_memory().percent
        # Evict oldest if cache full
        if len(self._cache) >= self._cache_size:

            evicted_path, _ = self._cache.popitem(last=False)
            # Optional: print(f"Evicted from cache due to size limit: {evicted_path}")

        # Cache miss: load from disk
        img = cv2.imread(str(path))
        if img is None:
            print(f"Warning: failed to read {path}")
            return None

        # Insert newly read image as most recently used
        self._cache[path] = img
        return img

    def _get_random_bake(self) -> Path:
        """Return a random bake image path."""
        if not self._bake_paths:
            raise FileNotFoundError(f"No bake images in {self.bake_dir}")
        return random.choice(self._bake_paths)

    @staticmethod
    def _merge_mask(
        shape: Tuple[int, int],
        segmentations: List[List[List[float]]]
        ) -> Tuple[np.ndarray, int, int, int, int]:
        """
        Combine multiple polygon segmentations into one mask and compute bounding box.
        Returns mask, x, y, w, h
        """
        mask = np.zeros(shape, dtype=np.uint8)
        points = []
        # Fill each polygon onto the mask
        for seg in segmentations:
            if isinstance(seg, list):
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)
                points.append(pts)
        if not points:
            # No valid segments, entire image
            return mask, 0, 0, shape[1], shape[0]
        all_pts = np.vstack(points)
        x, y, w, h = cv2.boundingRect(all_pts)
        return mask, x, y, w, h

    @staticmethod
    def _extract_and_crop(
        image: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mask to image and crop to bounding box.
        """
        extracted = cv2.bitwise_and(image, image, mask=mask)
        cropped_img = extracted[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        return cropped_img, cropped_mask

    @staticmethod
    def _add_padding(
        img: np.ndarray,
        mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random padding around the cropped image and mask.
        """
        h, w = img.shape[:2]
        max_edge = max(h, w)
        pad_extra = random.randint(0, max_edge)
        target = max_edge + pad_extra

        pad_vert = target - h
        pad_horiz = target - w
        top = random.randint(0, pad_vert)
        left = random.randint(0, pad_horiz)
        bottom = pad_vert - top
        right = pad_horiz - left

        img_padded = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        mask_padded = cv2.copyMakeBorder(
            mask, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
        return img_padded, mask_padded

    @staticmethod
    def _resize(
        img: np.ndarray,
        size: int
        ) -> np.ndarray:
        """Resize image to (size, size)."""
        return cv2.resize(img, (size, size))

    def _save(
        self,
        img: np.ndarray,
        name: str,
        folber:str = None
        ) -> None:
        """Save image with JPEG quality 95."""
        if folber is not None:
          dir = self.destination_dir / folber
        else:
          dir = self.destination_dir
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        out_path = dir / name
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def create_cut_images(
        self,
        min_area: int = 5000,
        show: bool = False,
        save: bool = True
        ) -> None:
        """
        Process annotations, extract objects, composite with random bakes,
        and optionally display or save results.
        """
        # Get annotation IDs for the target category
        ann_ids = self._coco.getAnnIds(catIds=[self.category_id])
        anns = self._coco.loadAnns(ann_ids)

        for idx, ann in enumerate(tqdm(anns, desc="Processing annotations")):
            if ann.get('area', 0) < min_area:
                continue

            # Load front image
            img_info = self._coco.loadImgs(ann['image_id'])[0]
            front_path = self.image_dir / img_info['file_name']
            front_img = self._read_image(front_path)
            if front_img is None:
                continue

            # Merge all segmentations into one mask
            mask, x, y, w, h = self._merge_mask(
                front_img.shape[:2],
                ann.get('segmentation', [])  # list of polygons
            )

            # Extract and crop object
            #mask1 = cv2.bitwise_not(mask)
            cropped, mask_cropped = self._extract_and_crop(front_img, mask, x, y, w, h)
            padded, mask_padded = self._add_padding(cropped, mask_cropped)

            # Composite with random bake background
            mask_inv = cv2.bitwise_not(mask_padded)
            bake_img = self._read_image(self._get_random_bake())
            if bake_img is None:
                continue
            bake_resized = self._resize(bake_img, padded.shape[0])
            bake_part = cv2.bitwise_and(bake_resized, bake_resized, mask=mask_inv)
            final = cv2.bitwise_or(bake_part, padded)

            # Show or save
            if show:
                rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 6))
                plt.imshow(rgb)
                plt.axis('off')
                plt.show()


            if save:
                name = f"cut_indx_{idx}_imageName_{img_info['file_name']}"
                from vision.Config import folber_name_cut_image
                self._save(final, name, folber_name_cut_image)


    def create_cut_box_images(
        self,
        min_area: int = 5000,
        max_area: int = 10000000,
        random_n:float = 0.2,
        show: bool = False,
        save: bool = True,
        random_size_box_cut = False
        ) -> None:

        """
        Process annotations, extract objects, composite with random bakes,
        and optionally display or save results.
        """
        # Get annotation IDs for the target category
        ann_ids = self._coco.getAnnIds(catIds=[self.category_id])
        anns = self._coco.loadAnns(ann_ids)
        self.bboxes_image :dict[str , List[List[float]]] = {}
        for idx, ann in enumerate(tqdm(anns, desc="Processing annotations")):

            # Load front image
            img_info = self._coco.loadImgs(ann['image_id'])[0]


            front_path = self.image_dir / img_info['file_name']
            front_img = self._read_image(front_path)
            if front_img is None:
                continue

            # Merge all segmentations into one mask

            bbox = ann.get('bbox', [])  # list of polygons
            x, y, w, h = [int(v) for v in bbox]



            if random_size_box_cut:
              random_size = random.randint(0, int(random_n * max(w,h)))
              size_cut  = max(w,h) + random_size

              #print('w')
              y1 = y - random.randint(0, max(int(w-h) , 0) + random_size)

              y = max(y1, 0)

              #print('h')
              x1 = x - random.randint(0, max(int(h-w), 0) + random_size)

              x = max(x1, 0)
              w = size_cut
              h = w

            # 4. برش مستقیم با استفاده از اسلایس NumPy
            y_end = min(y + h, front_img.shape[0])
            x_end = min(x + w, front_img.shape[1])
            cropped = front_img[y:y_end, x:x_end]

            # add bbox to list of box image
            self.bboxes_image.setdefault(img_info['file_name'], []).append(bbox)

            # Show or save
            if show:
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 6))
                plt.imshow(rgb)
                plt.axis('off')
                plt.show()

            if save and ann.get('area', 0) > min_area and ann.get('area', 0) < max_area :

                name = f"cut_area_{ann.get('area', 0)}_idx_{idx}_imageName_{img_info['file_name']}"
                from vision.Config import folber_name_box_image
                self._save(cropped, name , folber_name_box_image)

    def create_cut_box_bake(
        self,
        show: bool = False,
        save: bool = True,
        ) -> None:
      if len(self.bboxes_image) == 0 :
        print('Creat in  self.bboxes_image')
        print('you shoud de run function create_cut_box_images() ')
        return None

      for idx, img_name in enumerate(tqdm(self.bboxes_image.keys(), desc="Processing annotations")):

        bake_img = self._read_image(self.image_dir / img_name)
        if bake_img is None:
            continue




        h_img, w_img = bake_img.shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        for bbox in self.bboxes_image[img_name]:
            x, y, w, h = [int(v) for v in bbox]
            mask[y:y+h, x:x+w] = 255
            #bake_img[y:y+h, x:x+w] = 0
        random_img = self._read_image(self._get_random_bake())


        random_img = cv2.resize(random_img, (w_img, h_img))
        random_img = cv2.bitwise_and(random_img, random_img, mask=mask)
        mask_info = cv2.bitwise_not(mask)
        img_bake = cv2.bitwise_and(bake_img, bake_img, mask=mask_info)
        final = cv2.bitwise_or(random_img, img_bake, mask=None)


        # Show or save
        if show:
            rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb)
            plt.axis('off')
            plt.show()
        if save:
            name = f"bake_idx_{idx}_imageName_{img_name}"

            from vision.Config import folber_name_bake_image_Not
            self._save(final, name, folber_name_bake_image_Not)

    def cleanup(self):
        """
        Clear and delete all large internal attributes so that
        the garbage collector can reclaim their memory.
        """
        import gc

        # 1) LRU cache of loaded images
        if hasattr(self, '_cache'):
            self._cache.clear()    # حذف همه‌ی تصاویر از OrderedDict
            del self._cache        # حذف ارجاع به خودِ OrderedDict

        # 2) دیکشنری bboxes_image
        if hasattr(self, 'bboxes_image'):
            self.bboxes_image.clear()
            del self.bboxes_image

        # 3) لیست مسیرهای تصاویر bake
        if hasattr(self, '_bake_paths'):
            self._bake_paths.clear()
            del self._bake_paths

        # 4) شیء COCO (حاوی داده‌های آنتیشن)
        if hasattr(self, '_coco'):
            del self._coco

        # 5) حذف مقادیر مسیرها و پارامترهای دیگر (در صورت نیاز)
        for attr in ['image_dir', 'bake_dir', 'ann_file', 'destination_dir']:
            if hasattr(self, attr):
                delattr(self, attr)

        # 6) حذف تنظیمات کش/رم
        for attr in ['_cache_size', '_memory_threshold', '_use_ram', 'category_id']:
            if hasattr(self, attr):
                delattr(self, attr)

        # 7) حذف ماژول‌های import شده به عنوان attribute (در صورت بود)
        #    — معمولاً نیازی نیست مگر خودتان در __init__ ذخیره کرده باشید

        # 8) اجرای اجباری garbage collection
        gc.collect()

