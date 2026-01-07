import os
import random
import shutil

def copy_split_multiclass_dataset_to_dir(source_base_folder, output_base_folder, class_names, train_ratio=0.8, seed=None):
    """
    Randomly splits images from a source base folder (containing class subfolders)
    and outputs the 'train' and 'test' subfolders to a specific output directory.
    This function COPIES the files, leaving the original files untouched.

    Args:
        source_base_folder (str): The path to the main folder containing
                                  the class subfolders (e.g., 'dataset_x').
        output_base_folder (str): The path where the new 'train' and 'test'
                                  directories will be created.
        class_names (list): A list of strings, where each string is a class name.
        train_ratio (float): The proportion of images to be COPIED to the 'train' folder.
                             Must be between 0 and 1.
        seed (int, optional): A seed for the random number generator to ensure reproducibility.
    """
    if not os.path.isdir(source_base_folder):
        print(f"Error: Source folder not found at '{source_base_folder}'")
        return

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Create the root output directory if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Define the names for the output train and test folders
    train_base_folder = os.path.join(output_base_folder, 'train')
    test_base_folder = os.path.join(output_base_folder, 'test')

    print(f"Processing {len(class_names)} classes: {class_names}")

    # Iterate through each class name provided in the list
    for class_name in class_names:
        class_source_path = os.path.join(source_base_folder, class_name)
        
        # Check if the class source folder exists before proceeding
        if not os.path.isdir(class_source_path):
            print(f"Warning: Class folder '{class_source_path}' not found. Skipping.")
            continue
            
        # Create corresponding train and test subfolders for the current class
        train_class_path = os.path.join(train_base_folder, class_name)
        test_class_path = os.path.join(test_base_folder, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Get all image files for the current class
        all_files = [f for f in os.listdir(class_source_path) 
                     if os.path.isfile(os.path.join(class_source_path, f))]

        if not all_files:
            print(f"No images found in '{class_source_path}'. Skipping.")
            continue

        # Shuffle the list of files randomly
        random.shuffle(all_files)

        # Calculate the number of images for the training set
        num_train_images = int(len(all_files) * train_ratio)

        # Split the files into training and testing sets
        train_files = all_files[:num_train_images]
        test_files = all_files[num_train_images:]

        # Copy files to the respective train and test subfolders
        print(f"Class '{class_name}': Copying {len(train_files)} images to training set and {len(test_files)} to testing set.")
        
        for filename in train_files:
            src_path = os.path.join(class_source_path, filename)
            dst_path = os.path.join(train_class_path, filename)
            shutil.copy2(src_path, dst_path)  # Use copy2 instead of move
            
        for filename in test_files:
            src_path = os.path.join(class_source_path, filename)
            dst_path = os.path.join(test_class_path, filename)
            shutil.copy2(src_path, dst_path)  # Use copy2 instead of move

    print("Multiclass image copy complete.")