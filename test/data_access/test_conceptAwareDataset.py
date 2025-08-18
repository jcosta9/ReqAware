import logging
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image

import torch
import torchvision.transforms as transforms

# Import the class you want to test
from data_access.concepts import ConceptAwareDataset

# ---- Fixtures for Mocks and Test Data ----


@pytest.fixture
def mock_transform():
    """A mock transform function that returns a dummy tensor."""

    def transform_func(image):
        return torch.tensor([1, 2, 3])

    return transform_func


@pytest.fixture
def mock_image_data():
    """A mock object to simulate a PIL Image."""
    return Mock(spec=Image.Image)


@pytest.fixture
def concepts_csv_data():
    """Sample data for a concepts CSV file."""
    return """class_id,class_name,concept1,concept2,concept3
0,class_a,1,0,1
1,class_b,0,1,1
"""


@pytest.fixture
def setup_mock_filesystem(tmp_path, concepts_csv_data):
    """
    Sets up a mock directory structure with images and a concepts file.

    Returns a tuple of (root_dir, concepts_file_path, non_integer_dir).
    """
    root_dir = tmp_path / "dataset"
    root_dir.mkdir()

    # Create directories for two classes with integer names
    label_0_dir = root_dir / "0"
    label_0_dir.mkdir()
    (label_0_dir / "image1.png").touch()
    (label_0_dir / "image2.jpg").touch()

    label_1_dir = root_dir / "1"
    label_1_dir.mkdir()
    (label_1_dir / "image3.png").touch()
    (label_1_dir / "image4.jpg").touch()

    # Create a non-image file inside a label directory to be ignored
    (label_1_dir / "metadata.csv").touch()

    # Create a directory with a non-integer name to be ignored
    non_integer_dir = root_dir / "non_label_dir"
    non_integer_dir.mkdir()

    # Create a non-directory file in the root directory to be ignored
    (root_dir / "root_file.txt").touch()

    # Create a concepts file
    concepts_file = tmp_path / "concepts.csv"
    concepts_file.write_text(concepts_csv_data)

    return root_dir, concepts_file, non_integer_dir


# ---- Pytest Tests ----


def test_init_with_nonexistent_root_dir_raises_error(tmp_path, mock_transform):
    """Test that the constructor raises an error if the root directory does not exist."""
    non_existent_dir = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError, match="The directory .* does not exist."):
        ConceptAwareDataset(root_dir=non_existent_dir, transform=mock_transform)


@patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
def test_init_baseline_mode_no_concepts_file(
    mock_image_open, setup_mock_filesystem, mock_transform
):
    """Test the initialization for a baseline model (no concepts file)."""
    root_dir, _, _ = setup_mock_filesystem
    dataset = ConceptAwareDataset(root_dir=root_dir, transform=mock_transform)

    assert dataset.root_dir == root_dir
    assert dataset.transform == mock_transform
    assert dataset.concepts_file is None
    assert dataset.concepts is None
    assert len(dataset) == 4

    # Test __getitem__ for baseline model
    idx, img_transf, label = dataset[0]
    assert idx == 0
    assert torch.equal(img_transf, torch.tensor([1, 2, 3]))
    assert label == 0


@patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
def test_init_concept_aware_mode(
    mock_image_open, setup_mock_filesystem, mock_transform
):
    """Test the initialization for a concept-aware model."""
    root_dir, concepts_file, _ = setup_mock_filesystem
    dataset = ConceptAwareDataset(
        root_dir=root_dir, transform=mock_transform, concepts_file=concepts_file
    )

    assert dataset.concepts_file == concepts_file
    assert dataset.concepts is not None
    assert dataset.concepts_df is not None
    assert list(dataset.classes_names.values()) == ["class_a", "class_b"]
    assert list(dataset.concepts_names) == ["concept1", "concept2", "concept3"]
    assert len(dataset) == 4

    # Test __getitem__ for concept-aware model
    idx, img_transf, concepts_and_label = dataset[0]
    assert idx == 0
    assert torch.equal(img_transf, torch.tensor([1, 2, 3]))
    assert torch.equal(concepts_and_label[0], torch.tensor([1, 0, 1]))
    assert concepts_and_label[1] == 0


@patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
def test_init_with_nonexistent_concepts_file_raises_error(
    mock_image_open, setup_mock_filesystem, mock_transform
):
    """Test that the constructor raises an error if the concepts file does not exist."""
    root_dir, _, _ = setup_mock_filesystem
    non_existent_file = Path("non_existent_concepts.csv")

    with pytest.raises(FileNotFoundError):
        ConceptAwareDataset(
            root_dir=root_dir, transform=mock_transform, concepts_file=non_existent_file
        )


def test_get_images_labels_paths_logic(setup_mock_filesystem, mock_transform):
    """Test that get_images_labels_paths correctly finds images and ignores others."""
    root_dir, _, non_integer_dir = setup_mock_filesystem
    dataset = ConceptAwareDataset(root_dir=root_dir, transform=mock_transform)

    assert len(dataset.image_paths) == 4
    assert len(dataset.labels) == 4

    # Check that non-image files and non-integer directories were ignored
    for path in dataset.image_paths:
        assert "metadata.csv" not in path
        assert "root_file.txt" not in path
        assert str(non_integer_dir) not in path


def test_get_images_labels_paths_handles_non_integer_dirs(
    setup_mock_filesystem, mock_transform, caplog
):
    """Test that get_images_labels_paths logs a warning for non-integer directory names."""
    root_dir, _, _ = setup_mock_filesystem

    with caplog.at_level(logging.WARNING):
        ConceptAwareDataset(root_dir=root_dir, transform=mock_transform)

    assert "Skipping directory with non-integer name:" in caplog.text


def test_len_method_returns_correct_count(setup_mock_filesystem, mock_transform):
    """Test that the __len__ method returns the correct number of images."""
    root_dir, _, _ = setup_mock_filesystem
    dataset = ConceptAwareDataset(root_dir=root_dir, transform=mock_transform)

    assert len(dataset) == 4
