import pandas as pd
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset_by_keyword(csv_path, image_folder, keyword='', test_size=0.2, val_size=0.2, random_state=42):
    try:
        df = pd.read_csv(csv_path, sep='|', engine='python')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    df.columns = [col.strip() for col in df.columns]

    if 'comment' not in df.columns or 'image_name' not in df.columns:
        print("Missing required columns 'comment' or 'image_name' in CSV file.")
        return

    base_output_dir = '/home/cyt/cliperase/dataset/CIFAR100/data'
    
    directories = {
        'train_forget': os.path.join(base_output_dir, 'train', 'forget_set'),
        'train_retain': os.path.join(base_output_dir, 'train', 'retain_set'),
        'val_forget': os.path.join(base_output_dir, 'val', 'forget_set'),
        'val_retain': os.path.join(base_output_dir, 'val', 'retain_set'),
        'test_forget': os.path.join(base_output_dir, 'test', 'forget_set'),
        'test_retain': os.path.join(base_output_dir, 'test', 'retain_set')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    keyword_images = df[df['comment'].str.contains(keyword, case=False, na=False)]['image_name'].unique()

    forget_set = df[df['image_name'].isin(keyword_images)].copy()
    retain_set = df[~df['image_name'].isin(keyword_images)].copy()

    print(f"\nOriginal dataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Forget set samples: {len(forget_set)} (images containing keyword '{keyword}')")
    print(f"Retain set samples: {len(retain_set)}")

    forget_images = forget_set['image_name'].unique()
    if len(forget_images) > 0:
        forget_train_val, forget_test = train_test_split(
            forget_images, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        forget_train, forget_val = train_test_split(
            forget_train_val, test_size=val_ratio, random_state=random_state
        )
    else:
        forget_train, forget_val, forget_test = [], [], []

    retain_images = retain_set['image_name'].unique()
    if len(retain_images) > 0:
        retain_train_val, retain_test = train_test_split(
            retain_images, test_size=test_size, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        retain_train, retain_val = train_test_split(
            retain_train_val, test_size=val_ratio, random_state=random_state
        )
    else:
        retain_train, retain_val, retain_test = [], [], []

    print(f"\nDataset split statistics:")
    print(f"Forget set split:")
    print(f"  - Training set: {len(forget_train)} images")
    print(f"  - Validation set: {len(forget_val)} images") 
    print(f"  - Test set: {len(forget_test)} images")
    print(f"Retain set split:")
    print(f"  - Training set: {len(retain_train)} images")
    print(f"  - Validation set: {len(retain_val)} images")
    print(f"  - Test set: {len(retain_test)} images")

    def copy_images_and_save_csv(image_list, source_df, output_image_dir, output_csv_path):
        if len(image_list) == 0:
            return
            
        for image in image_list:
            try:
                shutil.copy(os.path.join(image_folder, image), output_image_dir)
            except Exception as e:
                print(f"Error copying image {image}: {e}")
        
        split_df = source_df[source_df['image_name'].isin(image_list)]
        split_df.to_csv(output_csv_path, index=False)

    print(f"\nProcessing forget set...")
    copy_images_and_save_csv(forget_train, forget_set, 
                           directories['train_forget'], 
                           os.path.join(directories['train_forget'], 'captions.csv'))
    
    copy_images_and_save_csv(forget_val, forget_set,
                           directories['val_forget'],
                           os.path.join(directories['val_forget'], 'captions.csv'))
    
    copy_images_and_save_csv(forget_test, forget_set,
                           directories['test_forget'],
                           os.path.join(directories['test_forget'], 'captions.csv'))

    print(f"Processing retain set...")
    copy_images_and_save_csv(retain_train, retain_set,
                           directories['train_retain'],
                           os.path.join(directories['train_retain'], 'captions.csv'))
    
    copy_images_and_save_csv(retain_val, retain_set,
                           directories['val_retain'],
                           os.path.join(directories['val_retain'], 'captions.csv'))
    
    copy_images_and_save_csv(retain_test, retain_set,
                           directories['test_retain'],
                           os.path.join(directories['test_retain'], 'captions.csv'))

    print(f"\nDataset split completed!")
    print(f"  Directory structure:")
    print(f"  {base_output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── forget_set/ (Training forget set: {len(forget_train)} images)")
    print(f"  │   └── retain_set/ (Training retain set: {len(retain_train)} images)")
    print(f"  ├── val/")
    print(f"  │   ├── forget_set/ (Validation forget set: {len(forget_val)} images)")
    print(f"  │   └── retain_set/ (Validation retain set: {len(retain_val)} images)")
    print(f"  └── test/")
    print(f"      ├── forget_set/ (Test forget set: {len(forget_test)} images)")
    print(f"      └── retain_set/ (Test retain set: {len(retain_test)} images)")
    
    total_images = len(forget_images) + len(retain_images)
    if total_images > 0:
        print(f"\n  Percentage statistics:")
        print(f"Training set: {(len(forget_train) + len(retain_train)) / total_images * 100:.1f}%")
        print(f"Validation set: {(len(forget_val) + len(retain_val)) / total_images * 100:.1f}%")
        print(f"Test set: {(len(forget_test) + len(retain_test)) / total_images * 100:.1f}%")


if __name__ == "__main__":
    csv_path = 'your path/to/captions.csv'  # Update this path to your actual CSV file
    image_folder = 'your path/to/image/folder'  # Update this path to your actual image folder
    base_output_dir = 'your path/to/output/folder'  # Update this path to your actual output folder

    split_dataset_by_keyword(
        csv_path, 
        image_folder, 
        keyword='mountain',
        test_size=0.1,
        val_size=0.2,
        random_state=42
    )
    
    