"""
Input: number of images, directory,
returns: array of 3d arrays (each d represeting a channel)
"""

import os

def fileHandling(directory_path):
    # Check if directory exists
    print("test")
    if not os.path.exists(directory_path):
        print("Directory not found.")
        return None
    print("test1")
    # Filter out files from directory
    file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    file_count = len(file_list)
    print("test2")
    if file_count == 0:
        print("Please paste your image in the image folder.")
        return None
    elif file_count == 1:
        return directory_path+file_list[0]
    elif file_count > 1:
        while True:
            file_name = input("Multiple images found. Please input the name of the image: ")
            file_path = os.path.join(directory_path, file_name)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                print("File exists.")
                return file_name
            else:
                print("File not found. Please try again.")