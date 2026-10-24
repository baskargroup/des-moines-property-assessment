import os
import xml.etree.ElementTree as ET
import pandas as pd
from glob import glob
import base64
from src.config import Config

class DataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = Config.DATA_DIR
        self.data_dir = data_dir
        self.annotations_dir = Config.ANNOTATIONS_DIR
        self.images_dir = Config.RAW_IMAGES_DIR 

    def load_annotations(self):
        """Parses all XML files in the annotations directory."""
        annotations_path = os.path.join(self.annotations_dir, "*.xml")
        xml_files = glob(annotations_path)
        
        # Debug output
        print(f"🔍 DataLoader debug:")
        print(f"   annotations_dir: {self.annotations_dir}")
        print(f"   annotations_dir exists: {os.path.exists(self.annotations_dir)}")
        print(f"   glob pattern: {annotations_path}")
        print(f"   XML files found: {len(xml_files)}")
        if xml_files:
            print(f"   Sample XML: {xml_files[0]}")
        
        data = []
        skipped_bad = 0
        skipped_no_good = 0
        processed_count = 0

        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            dataset_name = os.path.basename(xml_file).replace(".xml", "")

            for image in root.findall("image"):
                image_id = image.get("id")
                file_name = image.get("name")
                
                # Check for "bad" and "good" labels
                has_bad_label = False
                has_good_label = False
                
                for box in image.findall("box"):
                    label = box.get("label")
                    if label == "bad":
                        has_bad_label = True
                    elif label == "good":
                        has_good_label = True
                
                # Skip images with "bad" labels - only process images with "good" labels
                if has_bad_label:
                    skipped_bad += 1
                    continue  # Skip this image
                
                # Only process images that have "good" labels (or no bad labels)
                if not has_good_label:
                    skipped_no_good += 1
                    continue  # Skip images without "good" labels
                
                processed_count += 1
                
                # Construct full image path. 
                # Note: XML 'name' attribute might be like 'NHTyp1/ATT10035.jpg'
                # We need to join this with images_dir.
                # Handle both absolute and relative paths
                if os.path.isabs(self.images_dir):
                    full_image_path = os.path.join(self.images_dir, file_name)
                else:
                    # Relative to project root
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    full_image_path = os.path.join(base_dir, self.images_dir, file_name)

                expert_score = None
                
                # Extract expert score from boxes
                # Look for a box with label="expert_score" and extract its score attribute
                for box in image.findall("box"):
                    if box.get("label") == "expert_score":
                        # Score is stored in an attribute named "score"
                        for attr in box.findall("attribute"):
                            if attr.get("name") == "score":
                                score_text = attr.text
                                if score_text:
                                    try:
                                        expert_score = int(score_text.strip())
                                    except (ValueError, AttributeError):
                                        # If conversion fails, keep as string
                                        expert_score = score_text.strip()
                                break
                        # Break out of outer loop once we found expert_score
                        if expert_score is not None:
                            break
                
                data.append({
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_path": full_image_path,
                    "expert_score": expert_score
                })
        
        # Print filtering statistics
        print(f"\n📊 Image Filtering Statistics:")
        print(f"   Processed (with 'good' label): {processed_count}")
        print(f"   Skipped (has 'bad' label): {skipped_bad}")
        print(f"   Skipped (no 'good' label): {skipped_no_good}")
        print(f"   Total images in XML: {processed_count + skipped_bad + skipped_no_good}")

        return pd.DataFrame(data)

    @staticmethod
    def encode_image(image_path):
        """Encodes an image to base64."""
        if not os.path.exists(image_path):
            # Try fixing path if it's relative to the old 'Data' folder structure
            # e.g. if image_path is 'data/extractedimages/NHTyp1/img.jpg' but file is at
            # '/home/exouser/DSM.../Data/extractedimages/NHTyp1/img.jpg'
            # For now, we raise error or return None
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    df = loader.load_annotations()
    print(f"Loaded {len(df)} records.")
    print(df.head())

