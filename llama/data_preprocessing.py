import torch
from torch.utils.data import Dataset
import re
from PIL import Image
import pydicom

def create_report_path(row):
        base = 'mimic-cxr-reports/files/'
        p1 = 'p' + str(row['subject_id'])[:2]
        p2 = 'p' + str(row['subject_id'])
        s = 's' + str(row['study_id'])
        report_file = f"{base}{p1}/{p2}/{s}.txt"
        return report_file

    def create_image_path(row):
        base = 'physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
        p1 = 'p' + str(row['subject_id'])[:2]
        p2 = 'p' + str(row['subject_id'])
        s = 's' + str(row['study_id'])
        dcm = row['dicom_id'] + '.jpg' # can change to + '.dcm'
        return f"{base}{p1}/{p2}/{s}/{dcm}"

def my_collate_fn(batch):
        images = [item[0] for item in batch]
        reports = [item[1] for item in batch]
        images = torch.stack(images)  # This stacks the image tensors into a single tensor
        return images, reports

def preprocess_medical_report(text):
    # Remove section titles
    section_titles = [
        '(REASON FOR )?EXAMINATION',
        'HISTORY',
        'EXAMINATION',
        'INDICATION',
        'TECHNIQUE',
        'COMPARISON',
        'FINDINGS',
        'IMPRESSION']
    pattern = '|'.join([f'{title}:\\s*' for title in section_titles])
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = text.replace('FINAL REPORT', '')
    text = text.replace('_', ' ')

    # Replace 'h/o' with 'history of'
    text = re.sub(r'\bh/o\b', 'history of', text)

    # Replace 'r/o' with 'rule out'
    text = re.sub(r'\br/o\b', 'rule out', text)

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # replace medical abbreviations
    text = re.sub(r'\bR>L\b', 'right greater than left', text)
    text = re.sub(r'\bL>R\b', 'left greater than right', text)
    text = text.lower()
    text = text.replace('none', '')

    # Additional step: Remove extra whitespaces
    text = ' '.join(text.split()).strip()

    return text

class MIMIC_CXR(Dataset):
        def __init__(self, metadata_df, augment=None, img_depth=3, image_format='jpg'): # can change to jpg format
            self.metadata_df = metadata_df
            self.augment = augment
            self.img_depth = img_depth
            self.image_format = image_format

        def __len__(self):
            return len(self.metadata_df)

        def __getitem__(self, index):
            # Getting image path and loading image based on format
            file = self.metadata_df.iloc[index]['image_path']

            if self.image_format == 'dcm':
                dicom_file = pydicom.dcmread(file)
                imageData = dicom_file.pixel_array
                imageData = Image.fromarray(imageData).convert("L")
            else:  # Assuming it's a format readable by PIL (like jpg)
                imageData = Image.open(file)
                if self.img_depth == 1:
                    imageData = imageData.convert('L')
                else:
                    imageData = imageData.convert('RGB')

            # Applying augmentations if any
            if self.augment:
                imageData = self.augment(imageData)

            report_path = self.metadata_df.iloc[index]['report_path']
            try:
                with open(report_path, 'r', encoding='utf-8') as report_file:
                    report_text = report_file.read()
                    report_text = preprocess_medical_report(report_text)
            except FileNotFoundError:
                print(f"Report file not found for index {index}: {report_path}")
                report_text = ""
            except Exception as e:
                print(f"An error occurred while reading the report for index {index}: {e}")
                report_text = ""

            return imageData, report_text