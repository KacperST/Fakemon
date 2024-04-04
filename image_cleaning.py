import os
from typing import Tuple
import cv2
import numpy as np
from tqdm import tqdm

def check_size(dataset_path: str) -> None:
    for file in tqdm(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, file)
        img = cv2.imread(file_path)

        width, height, _ = img.shape

        if width < 256 or height < 256:
            os.remove(file_path)

    print(f"\nChecking size is done.\nRemaining images: {len(os.listdir(dataset_path))}\n")

def change_white_background_to_black(dataset_path: str, output_dir_path:str) -> None:
    # create output directory if it does not exist
    if not os.path.exists(output_dir_path):
        os.makedirs(os.path.join(os.getcwd(), output_dir_path))
    for file in tqdm(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(I)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(I, mask)
        file_name, file_extension = os.path.splitext(file)
        new_file_name = file_name + f"_black_background" + ".jpg"
        output_path = os.path.join(output_dir_path, new_file_name)
        cv2.imwrite(output_path, result)
    
    print("\nBackgrounds changed to black.\n")
       

def resize_images2(dataset_path: str,output_dir_path:str, image_size:Tuple[int, int] =(256, 256)) -> None:
    for file in tqdm(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        I_resized = cv2.resize(I, image_size)
        file_name, file_extension = os.path.splitext(file)
        new_file_name = file_name + file_extension
        output_path = os.path.join(output_dir_path, new_file_name)
        cv2.imwrite(output_path, I_resized)
    
    print("\nImages resized.\n")

def change_contrast(dataset_path: str, output_dir_path:str, alpha: float, beta: float) -> None:
    for file in tqdm(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        adjusted = cv2.convertScaleAbs(I, alpha=alpha, beta=beta)
        file_name, file_extension = os.path.splitext(file)
        new_file_name = file_name + f"_contrast" + file_extension
        output_path = os.path.join(output_dir_path, new_file_name)
        cv2.imwrite(output_path, adjusted)

    print("\nContrast added.\n")


def add_reflection(dataset_path: str, output_dir_path:str) -> None:
    for file in tqdm(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        I_reflected = cv2.flip(I, 1)  # Flip horizontally
        file_name, file_extension = os.path.splitext(file)
        new_file_name = file_name + f"_reflected" + file_extension
        output_path = os.path.join(output_dir_path, new_file_name)
        cv2.imwrite(output_path, I_reflected)
    
    print("\nReflections added.\n")


def add_noise2(dataset_path: str, output_dir_path:str) -> None:
    for file in tqdm(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        row, col, ch = I.shape
        mean = 0
        std_dev = 5
        gauss = np.random.normal(mean, std_dev, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = I + gauss
        file_name, file_extension = os.path.splitext(file)
        new_file_name = file_name + f"_noisy" + file_extension
        output_path = os.path.join(output_dir_path, new_file_name)
        cv2.imwrite(output_path, noisy)
    
    print("\nNoise added.\n")

def add_negative(dataset_path: str, output_dir_path: str, image_size:Tuple[int, int] =(256, 256)) -> None:
    def filter_jpg(file: str):
        if file.split('.')[1].lower() == 'jpg':
            return True
        else:
            return False
    
    jpg_images = list(filter(filter_jpg, os.listdir(dataset_path)))
    for file in tqdm(jpg_images):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        I_r = cv2.resize(I, image_size)
        I_neg = abs(255-I_r)
        file_name, file_extension = os.path.splitext(file)
        new_file_name = file_name + f"_negative" + file_extension
        output_path = os.path.join(output_dir_path, new_file_name)
        cv2.imwrite(output_path, I_neg)

    print("\nNegative added.\n")

def pipeline(dataset_path: str, output_dir_path:str) -> None:
    check_size(dataset_path)
    change_white_background_to_black(dataset_path, output_dir_path)
    resize_images2(output_dir_path, output_dir_path, (256, 256))
    add_reflection(output_dir_path, output_dir_path)
    change_contrast(output_dir_path, output_dir_path, 1.3, 0)
    add_noise2(output_dir_path, output_dir_path)
    add_negative(dataset_path, output_dir_path)



def main() -> None:
    current_dir = os.getcwd()
    dataset_path = f"{current_dir}/pokemons"
    output_dir = f"{current_dir}/pokemons_noise"
    pipeline(dataset_path, output_dir)
    
    
if __name__ == "__main__":
    main()
    
