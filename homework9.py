'''

File name: get-measuraments.py 
Description: This script matchs the images loaded using shi-Tomasi and sift 

Author(s): Victor Santiago Solis Garcia - Jonathan Ariel Valadez Saldaña


Creation date: 04/26/2024

Usage example:  python homework09.py --input_images box.png box_in_scene.png 
'''


import cv2
import numpy as np
import argparse
from typing import Tuple

def parse_user_data()->argparse:
    """
    Function to input the user data 
    
    Parameter(s):    None

    Returns:       args(argparse): argparse object with the user info
    """
    parser = argparse.ArgumentParser(description='Feature matching between two images.')
    parser.add_argument('--input_images', nargs=2, required=True,
                        help='Input images for feature matching')
    args = parser.parse_args()
    return args

def load_images(image_paths: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads two images from the specified paths 
    
    Parameter(s):    image_paths(str): array with the image paths 

    Returns:       img1,img2(ndarray): both loaded images
    """
    img1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: One of the images failed to load. Check the file paths.")
        exit()
    return img1, img2

def process_images(img1:np.ndarray, img2:np.ndarray)->np.ndarray:
    """
    Process two images to obtain the corners and match the images 

    Parameter(s): img1(ndarray),img2(ndarray): loaded images
    

    Returns: match_img(ndarray):An image with the matched points
    """
    corners1 = cv2.goodFeaturesToTrack(img1, 500, 0.01, 10)
    corners2 = cv2.goodFeaturesToTrack(img2, 500, 0.01, 10)

    kp1 = [cv2.KeyPoint(x=corner[0][0], y=corner[0][1], size=20) for corner in corners1]
    kp2 = [cv2.KeyPoint(x=corner[0][0], y=corner[0][1], size=20) for corner in corners2]

    sift = cv2.SIFT_create()
    _, des1 = sift.compute(img1, kp1)
    _, des2 = sift.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    match_img = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return match_img

def display_images(match_img:np.ndarray)->None:
    """
    Display image 

    Parameter(s): match_img(ndarray): image with the matched points
    

    Returns: None
    """
    cv2.imshow('Matches', match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pipeline()->None:
    args = parse_user_data()
    image_paths = args.input_images
    img1, img2 = load_images(image_paths)
    match_img = process_images(img1, img2)
    display_images(match_img)

if __name__ == "__main__":
    pipeline()
