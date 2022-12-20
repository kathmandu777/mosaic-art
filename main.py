from PIL import Image
import cv2
import numpy as np
import glob
from scipy import spatial
from tqdm import tqdm
from PIL.Image import Image as ImageType
import argparse
import pickle
import os


def get_represent_color(pil_image: ImageType) -> np.ndarray:
    """Get represent color of image

    (e.g.) [112.36366956 119.19114827 127.05403072]
    """
    return pil2cv(pil_image).mean(axis=0).mean(axis=0)


def crop_center(pil_image: ImageType, crop_width: int, crop_height: int) -> ImageType:
    img_width, img_height = pil_image.size
    return pil_image.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


def cv2pil(cv2_image: np.ndarray) -> ImageType:
    """Convert cv2 image to PIL image"""
    new_image = cv2_image.copy()
    if new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(new_image)


def pil2cv(pil_image: ImageType) -> np.ndarray:
    """Convert PIL image to cv2 image"""
    new_image = np.array(pil_image, dtype=np.uint8)

    if new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def main(
    target_image_path: str,
    tile_size: int,
    image_dir: str,
    pop: bool,
    zoom: int,
):
    THRESHOLD_IMAGES_NUM = 2000

    # Mosaic Art Info
    print("\n===============================")
    print("Mosaic Art Info")
    print("===============================")
    target_image = cv2.imread(target_image_path)
    print(f"Target image shape: {target_image.shape}")

    target_image = target_image[
        : target_image.shape[0] // tile_size * tile_size,
        : target_image.shape[1] // tile_size * tile_size,
    ]
    print(
        f"Needed images: {target_image.shape[0] // tile_size} x {target_image.shape[1] // tile_size} = {target_image.shape[0] // tile_size * target_image.shape[1] // tile_size}"
    )

    print(
        f"Output image size (zoom: {zoom}): {target_image.shape[0] * zoom} x {target_image.shape[1] * zoom}"
    )

    confirm = input("Continue? [Y/n]: ")
    if confirm == "n":
        exit()

    # Load images and get represent color
    print("\n===============================")
    print("Loading images...")
    print("===============================")

    image_paths = list(
        set(
            glob.glob(image_dir + "/**/*.jpg", recursive=True)
            + glob.glob(image_dir + "/**/*.png", recursive=True)
            + glob.glob(image_dir + "/**/*.JPG", recursive=True)
        )
    )
    if os.path.isfile("image_paths.pkl") and os.path.isfile(
        "image_represent_colors.pkl"
    ):
        with open("image_paths.pkl", "rb") as f:
            pickle_image_paths = pickle.load(f)
        if set(pickle_image_paths) == set(image_paths):
            with open("image_paths.pkl", "rb") as f:
                image_paths = pickle.load(f)
            with open("image_represent_colors.pkl", "rb") as f:
                image_represent_colors = pickle.load(f)
        else:
            image_represent_colors = []
            for image_path in tqdm(image_paths):
                image = Image.open(image_path).convert("RGB")
                image_represent_colors.append(
                    get_represent_color(
                        crop_center(image, min(image.size), min(image.size))
                    )
                )
            with open("image_paths.pkl", "wb") as f:
                pickle.dump(image_paths, f)
            with open("image_represent_colors.pkl", "wb") as f:
                pickle.dump(image_represent_colors, f)
    else:
        image_represent_colors = []
        for image_path in tqdm(image_paths):
            image = Image.open(image_path).convert("RGB")
            image_represent_colors.append(
                get_represent_color(
                    crop_center(image, min(image.size), min(image.size))
                )
            )
        with open("image_paths.pkl", "wb") as f:
            pickle.dump(image_paths, f)
        with open("image_represent_colors.pkl", "wb") as f:
            pickle.dump(image_represent_colors, f)

    print(f"Loaded images: {len(image_represent_colors)}")

    # Create mosaic art
    print("\n===============================")
    print("Creating mosaic...")
    print("===============================")
    original_image_represent_colors = image_represent_colors.copy()
    original_image_paths = image_paths.copy()

    generated_image = np.zeros(
        (target_image.shape[0] * zoom, target_image.shape[1] * zoom, 3),
        dtype=np.uint8,
    )
    for i in tqdm(range(target_image.shape[0] // tile_size)):  # height
        for j in range(target_image.shape[1] // tile_size):  # width
            if len(image_represent_colors) < THRESHOLD_IMAGES_NUM:
                image_represent_colors = original_image_represent_colors.copy()
                image_paths = original_image_paths.copy()
            tree = spatial.KDTree(image_represent_colors)
            target_tile = target_image[
                i * tile_size : (i + 1) * tile_size,
                j * tile_size : (j + 1) * tile_size,
            ]
            _, index = tree.query(get_represent_color(cv2pil(target_tile)))
            image = Image.open(image_paths[index]).convert("RGB")
            generated_image[
                i * tile_size * zoom : (i + 1) * tile_size * zoom,
                j * tile_size * zoom : (j + 1) * tile_size * zoom,
            ] = pil2cv(
                crop_center(image, min(image.size), min(image.size)).resize(
                    (tile_size * zoom, tile_size * zoom)
                )
            )

            if pop:
                image_represent_colors.pop(index)
                image_paths.pop(index)

    # Save mosaic art
    print("\n===============================")
    print("Saving image...")
    print("===============================")
    cv2.imwrite("result_100percent.jpg", generated_image)
    for i in range(1, 10):
        ratio = i / 10
        image = cv2.addWeighted(
            generated_image,
            ratio,
            cv2.resize(target_image, generated_image.shape[:2][::-1]),
            1 - ratio,
            0,
        )
        cv2.imwrite(f"result_{i*10}percent.jpg", image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mosaic Art Generator")
    parser.add_argument("--target_image", type=str, help="Target image path")
    parser.add_argument("--tile_size", type=int, help="Tile size", default=100)
    parser.add_argument(
        "--images_path",
        type=str,
        help="Image folder path",
    )
    parser.add_argument(
        "--not_pop", help="Not to pop image from list", action="store_true"
    )
    parser.add_argument("--zoom", type=int, help="Zoom", default=1)

    args = parser.parse_args()
    assert args.tile_size > 0
    assert args.zoom >= 1

    main(
        args.target_image, args.tile_size, args.images_path, not args.not_pop, args.zoom
    )
