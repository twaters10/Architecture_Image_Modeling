import os
import sys
# Get the path to the directory you want to import from
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from house_classification.utils.config import DATA_PATH, house_styles, TRAINING_DATA_PATH
from house_classification.utils.yolo_data_storage import copy_split_multiclass_dataset_to_dir

if __name__ == '__main__':
    source_folder = DATA_PATH
    output_folder = TRAINING_DATA_PATH
    my_classes = house_styles

    # Create dummy folders and files for demonstration
    for class_name in my_classes:
        class_path = os.path.join(source_folder, class_name)
        os.makedirs(class_path, exist_ok=True)
        for i in range(1, 11):
            with open(os.path.join(class_path, f'{class_name}_{i}.jpg'), 'w') as f:
                f.write(f"This is a {class_name} file.")
    
    # Call the updated function
    copy_split_multiclass_dataset_to_dir(
        source_base_folder=source_folder,
        output_base_folder=output_folder,
        class_names=my_classes,
        train_ratio=0.8,
        seed=42
    )