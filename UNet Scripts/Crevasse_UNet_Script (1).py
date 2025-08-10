import os, sys, glob, random, io
import numpy as np
import rasterio
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output

sys.path.append('jupyter-kernels/UNet_Scripts/')
from tfUNet import unet_model_custom
import metrics_custom

# ─────────────── config ───────────────
ROOT          = "/home/common/HolschuhLab/Data/Antarctica_UNet_Training/"
TRAIN_GLOB    = os.path.join(ROOT, "training090422", "*.tif")
DATA_SUFFIX   = ".tif"
MASK_SUFFIX   = "_mask.tif"
TARGET_SIZE   = 512
BATCH_SIZE    = 8
BUFFER_SIZE   = 1_000

# ─────────── raster helper ────────────
def _read_tiff_from_path(path_bytes):
    if isinstance(path_bytes, np.ndarray):
        path_bytes = path_bytes.item()
    path = path_bytes.decode()
    with rasterio.open(path) as ds:
        arr = ds.read()          # (C,H,W)
    arr = np.transpose(arr, (1, 2, 0))         # (H,W,C)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr.astype(np.uint16)

def decode_tiff(path_tensor):
    img = tf.numpy_function(_read_tiff_from_path, [path_tensor], tf.uint16)
    img.set_shape([None, None, None])
    return img

# ─────────── dataset loader ───────────
def load_image_mask(img_path, mask_path):
    img  = decode_tiff(img_path)[..., 0]
    mask = decode_tiff(mask_path)[..., 0]

    img  = tf.cast(img, tf.float32)
    minv = tf.reduce_min(img)
    maxv = tf.reduce_max(img)
    rangev = tf.maximum(maxv - minv, 1.0)
    img   = (img - minv) / rangev

    mask  = tf.cast(mask > 0, tf.float32)

    img   = img[..., tf.newaxis]
    mask  = mask[..., tf.newaxis]

    # pad to 512²
    img   = tf.image.pad_to_bounding_box(img,  0, 0, TARGET_SIZE, TARGET_SIZE)
    mask  = tf.image.pad_to_bounding_box(mask, 0, 0, TARGET_SIZE, TARGET_SIZE)

    # weight = 0 on padded border
    weight = tf.where(tf.equal(img, 0.0), 0.0, 1.0)

    return img, mask, weight

# ───────── augmentation (image, mask, weight) ─────────
def Augment():
    def _aug(img, mask, weight):
        # geometric
        if tf.random.uniform([]) > 0.5:
            img  = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
            weight = tf.image.flip_left_right(weight)

        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img  = tf.image.rot90(img, k)
        mask = tf.image.rot90(mask, k)
        weight = tf.image.rot90(weight, k)

        # photometric (image only)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, mask, weight
    return _aug

# ─────────── build file lists ──────────
image_paths = sorted(p for p in glob.glob(TRAIN_GLOB)
                     if not p.endswith(MASK_SUFFIX))
mask_paths  = [p.replace(DATA_SUFFIX, MASK_SUFFIX) for p in image_paths]

train_imgs, val_imgs, train_msk, val_msk = train_test_split(
    image_paths, mask_paths, test_size=0.20, random_state=42, shuffle=True)

print(f"Train images: {len(train_imgs)}   Val images: {len(val_imgs)}")

# ─────────── tf.data pipelines ─────────
train_ds = (tf.data.Dataset
            .from_tensor_slices((train_imgs, train_msk))
            .repeat()
            .shuffle(BUFFER_SIZE, seed=42, reshuffle_each_iteration=True)
            .map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
            .map(Augment(),      num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds   = (tf.data.Dataset
            .from_tensor_slices((val_imgs, val_msk))
            .map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

# ───────────── loss functions ──────────
def dice_loss(y_true, y_pred, smooth=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.nn.sigmoid(y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f)
    denom    = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.0*inter + smooth) / (denom + smooth)

POS_WEIGHT = 8.0

def weighted_bce(y_true, y_pred):
    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=tf.cast(y_true, tf.float32),
        logits=y_pred,
        pos_weight=POS_WEIGHT
    )
    return tf.reduce_mean(loss)

def combo_loss(y_true, y_pred):
    return 0.5 * weighted_bce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# ──────────── build model ──────────────
model = unet_model_custom()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=combo_loss,
              metrics=['accuracy', metrics_custom.DiceCoefficient()],
              weighted_metrics = []
             )

# ─────────── callbacks ───────────
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_dice_coef', mode='max',
    factor=0.5, patience=4, min_lr=1e-6)

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_dice_coef', mode='max',
    patience=10, restore_best_weights=True)

# ─────────── training ───────────
EPOCHS = 100
STEPS  = len(train_imgs) // BATCH_SIZE

history = model.fit(
    train_ds.map(lambda x, y, w: (x, y, w)),
    epochs=EPOCHS,
    steps_per_epoch=STEPS,
    validation_data=val_ds.map(lambda x, y, w: (x, y, w)),
    callbacks=[reduce_lr],
    verbose=1
)

# ───── save model & curves ─────
model.save('jupyter-kernels/UNet_Scripts/unet_crevasse.keras')


#==================================================
#================ Image ===========================
#==================================================

history_dict = history.history

train_epochs_loss  = range(1, len(history_dict['loss']) + 1)
val_epochs_loss    = range(1, len(history_dict['val_loss']) + 1)

train_epochs_acc   = range(1, len(history_dict['accuracy']) + 1)
val_epochs_acc     = range(1, len(history_dict['val_accuracy']) + 1)

train_epochs_dice  = range(1, len(history_dict['dice_coef']) + 1)
val_epochs_dice    = range(1, len(history_dict['val_dice_coef']) + 1)


save_dir  = 'jupyter-kernels/UNet_Scripts'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'Crevasse_unettraining_history.png')

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(train_epochs_loss, history_dict['loss'],  'r-',  label='Train Loss')
plt.plot(val_epochs_loss,   history_dict['val_loss'], 'b-',  label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_epochs_acc,  history_dict['accuracy'],     'g-',  label='Train Acc')
plt.plot(val_epochs_acc,    history_dict['val_accuracy'], 'm-',  label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(train_epochs_dice, history_dict['dice_coef'],     'c-',  label='Train Dice')
plt.plot(val_epochs_dice,   history_dict['val_dice_coef'], 'y-',  label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(save_path, facecolor='white')
plt.show()