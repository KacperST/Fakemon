import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def move_images_to_different_director(image: np.array, output_dir: str, file:str) -> None:
    output_path = os.path.join(output_dir, file)
    cv2.imwrite(output_path, image)
         
         
def resize_images():
    current_dir = os.getcwd()
    dataset_path = f"{current_dir}/pokemons"
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        I = cv2.imread(file_path)
        I_resized = cv2.resize(I, (512, 512))
        output_dir = f"{current_dir}/pokemon_resized_512"
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

            move_images_to_different_director(result, f"{current_dir}/pokemons_black_background", file)
            
def main():
    change_white_background_to_black()  
    
if __name__ == "__main__":
    main()
    