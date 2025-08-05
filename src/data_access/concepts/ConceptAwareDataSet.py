import os
import logging
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ConceptAwareDataset(Dataset):
    """
    Returns a compatible Torch Dataset object that allows for concept annotations
    """

    def __init__(
        self,
        root_dir,
        transform,
        concepts_file=None,  # if no file is passed, the dataloader works for baseline models only.
    ):

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # concept-aware dataset parameters
        self.concepts_file = concepts_file
        self.concepts = None
        self.concepts_df = None
        self.classes_names = None
        self.concepts_names = None

        if not root_dir.exists():
            logging.error(f"[DATA ACCESS] The directory {root_dir} does not exist.")
            raise FileNotFoundError(f"The directory {root_dir} does not exist.")

        self.get_images_labels_paths()

        if not concepts_file:  # baseline model loads only labels
            logging.info(
                f"[DATA ACCESS] No concepts file were given. Loading standard dataloader."
            )
            return

        # if concepts_file is given, the dataset is concept-aware
        # concept-aware dataset loads concepts, labels and images
        logging.info(
            f"[DATA ACCESS] Concepts file {self.concepts_file} was given. Loading CBM dataloader, i.e., Input-concepts-labels."
        )

        # loading concepts per class dict
        try:
            self.concepts_df = pd.read_csv(self.concepts_file).set_index("class_id")
            self.concepts = {
                class_id: torch.tensor(concepts_list, dtype=torch.float32)
                for class_id, concepts_list in zip(
                    range(len(self.concepts_df)), self.concepts_df.iloc[:, 1:].values
                )
            }
        except FileNotFoundError as e:
            logging.error(
                f"[DATA ACCESS] The concepts file {self.concepts_file} does not exist."
            )
            raise e

        # saving classes and concepts names for reference
        self.classes_names = self.concepts_df.class_name.to_dict()
        self.concepts_names = self.concepts_df.columns[1:]

        logging.info(f"[DATA ACCESS] Concepts list: {self.concepts_names}")

    def get_images_labels_paths(self) -> None:
        """
        Get all images and labels from the root directory.
        The root directory should contain subdirectories for each class.
        Each subdirectory should contain images of that class.
        """
        root_path = Path(self.root_dir)
        for label_dir in root_path.iterdir():
            if label_dir.is_dir():
                try:
                    label = int(label_dir.name)
                except ValueError:
                    logging.warning(
                        f"Skipping directory with non-integer name: {label_dir}"
                    )
                    continue

                for image_path in label_dir.iterdir():
                    # We can check the file extension directly
                    if image_path.is_file() and image_path.suffix.lower() not in [
                        ".csv",
                        ".txt",
                    ]:
                        self.image_paths.append(str(image_path))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            img_transf = self.transform(image)

        if not self.concepts:  # standard model
            return idx, img_transf, label

        # concept-aware model
        concepts = self.concepts[label]
        return idx, img_transf, (concepts, label)
