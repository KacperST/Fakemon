import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def move_images_to_different_director(image: np.array, output_dir: str, file:str, word: str) -> None:
    file_name, file_extension = os.path.splitext(file)
    new_file_name = file_name + f"_{word}" + file_extension
    output_path = os.path.join(output_dir, new_file_name)
    cv2.imwrite(output_path, image)
         
         
def resize_images():
    current_dir = os.getcwd()
    dataset_path = f"{current_dir}/pokemons"
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        I_resized = cv2.resize(I, (512, 512))
        output_dir = f"{current_dir}/pokemons_resized_512"
        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, I_resized)


def change_white_background_to_black():
    current_dir = os.getcwd()
    dataset_path = f"{current_dir}/pokemons"
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if file.endswith(".jpg"):
            image = cv2.imread(file_path)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask = np.zeros_like(image)

            cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

            result = cv2.bitwise_and(image, mask)

            move_images_to_different_director(result, f"{current_dir}/pokemons_black_background", file, "")
            
            
def mirror_reflection():
    current_dir = os.getcwd()
    dataset_path = f"{current_dir}/pokemons_black_background"
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if file.endswith(".jpg"):
            image = cv2.imread(file_path)
            image = cv2.flip(image, 1)  # Flip horizontally
            move_images_to_different_director(image, f"{current_dir}/pokemons_mirror_reflection", file, "reflected")

            
def add_vary_contrast():
    current_dir = os.getcwd()
    dataset_path = f"{current_dir}/pokemons_mirror_reflection"
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if file.endswith(".jpg"):
            image = cv2.imread(file_path)
            alpha = 1.3  # Contrast control (1.0-3.0)
            beta = 0  # Brightness control (0-100)
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            move_images_to_different_director(adjusted, f"{current_dir}/pokemons_contrast", file, "contrast")            

def add_noise():
    current_dir = os.getcwd()
    dataset_path = f"{current_dir}/pokemons_contrast"
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if file.endswith(".jpg"):
            image = cv2.imread(file_path)
            row, col, ch = image.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            move_images_to_different_director(noisy, f"{current_dir}/pokemons_noise", file, "noise")


def main():
    add_noise()
    
if __name__ == "__main__":
    main()
    