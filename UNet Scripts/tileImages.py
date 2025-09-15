def tile_image(img, tile_size=512):
    """
    Splits a 2D/3D image into non-overlapping tiles of shape (tile_size, tile_size, C).
    Returns list of (tile, (row_offset, col_offset)).
    """
    H, W = img.shape[:2]
    tiles = []
    for r in range(0, H, tile_size):
        for c in range(0, W, tile_size):
            tile = img[r:r+tile_size, c:c+tile_size]
            # pad if necessary
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0,0)), mode="constant")
            tiles.append((tile, (r, c)))
    return tiles, (H, W)

def predict_large_image(model, img, tile_size=512):
    """
    Predict mask for a full image by tiling into patches.
    """
    tiles, (H, W) = tile_image(img, tile_size)
    full_pred = np.zeros((H, W), dtype=np.float32)

    for tile, (r, c) in tiles:
        tile_in = tile[..., np.newaxis] if tile.ndim == 2 else tile
        tile_in = tile_in.astype(np.float32)
        # normalize like training
        tile_in = (tile_in - tile_in.min()) / max(tile_in.max()-tile_in.min(), 1e-6)
        tile_in = np.expand_dims(tile_in, 0)  # add batch dim

        pred = model.predict(tile_in, verbose=0)[0, ..., 0]  # (512,512)

        h = min(512, H - r)
        w = min(512, W - c)
        full_pred[r:r+h, c:c+w] = pred[:h, :w]

    return full_pred

import rasterio

def save_prediction_as_tiff(output_path, pred, ref_path):
    """
    Save predicted mask as GeoTIFF aligned with the reference image.
    """
    with rasterio.open(ref_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred.astype(np.float32), 1)

model = tf.keras.models.load_model(
    "jupyter-kernels/UNet_Scripts/unet_crevasse.keras",
    custom_objects={"combo_loss": combo_loss,
                    "DiceCoefficient": metrics_custom.DiceCoefficient()}
)

# run prediction
input_tiff = "/path/to/antarctica_scene.tif"
with rasterio.open(input_tiff) as ds:
    img = ds.read()  # (C,H,W)
    img = np.transpose(img, (1,2,0))  # (H,W,C)

pred_mask = predict_large_image(model, img, tile_size=512)

# save stitched prediction
save_prediction_as_tiff("antarctica_mask_pred.tif", pred_mask, input_tiff)
