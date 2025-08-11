import pytest
import torch
import torchvision.transforms as transforms
from unittest.mock import Mock, patch

# Assuming the factory and dataset classes are in these locations
from data_access.datasets.GTSRBFactory import GTSRBFactory, GTSRB_transform
from data_access.data_factory import DatasetFactory
from data_access.concepts.ConceptAwareDataSet import ConceptAwareDataset

# ---- Mocks and Fixtures for Testing ----

class MockConfig:
    """A mock configuration object for testing."""
    def __init__(self, data_path, concepts_file=None):
        self.data_path = data_path
        self.val_split = 0.2
        self.seed = 42
        self.concepts_file = concepts_file

# --- Fixtures to set up test environment ---

@pytest.fixture
def mock_root_dirs(tmp_path):
    """Creates mock data directories for the factory to find."""
    data_path = tmp_path / "data"
    training_path = data_path / "training"
    test_path = data_path / "test"
    training_path.mkdir(parents=True)
    test_path.mkdir(parents=True)
    return data_path, training_path, test_path

@pytest.fixture
def mock_config(mock_root_dirs):
    """Provides a mock config object with a temporary data path."""
    data_path, _, _ = mock_root_dirs
    concepts_file_path = data_path / "concepts.csv"
    concepts_file_path.touch()
    
    return MockConfig(data_path=data_path, concepts_file=concepts_file_path)

@pytest.fixture
def factory(mock_config):
    """Provides a GTSRBFactory instance for each test."""
    return GTSRBFactory(mock_config)

# ---- Pytest Tests ----

def test_inheritance():
    """Test that GTSRBFactory correctly inherits from DatasetFactory."""
    assert issubclass(GTSRBFactory, DatasetFactory)

def test_init(factory, mock_config):
    """Test that the constructor correctly initializes the factory."""
    assert isinstance(factory, GTSRBFactory)
    assert factory.config is mock_config

# This test already has a fully working patch for random_split
@patch('data_access.datasets.GTSRBFactory.random_split')
@patch('data_access.datasets.GTSRBFactory.ConceptAwareDataset', return_value=Mock(spec=ConceptAwareDataset, __len__=Mock(return_value=100)))
def test_load_datasets_calls_correct_classes(MockConceptAwareDataset, mock_random_split, factory, mock_root_dirs, mock_config):
    """Test that load_datasets instantiates ConceptAwareDataset and calls random_split."""
    data_path, training_path, test_path = mock_root_dirs

    # A mock dataset with a specified length is returned by the patch
    mock_random_split.return_value = [
        Mock(spec=ConceptAwareDataset, __len__=Mock(return_value=80)),
        Mock(spec=ConceptAwareDataset, __len__=Mock(return_value=20))
    ]

    result = factory.load_datasets()

    # Assert that ConceptAwareDataset was called correctly for train and test
    assert MockConceptAwareDataset.call_count == 2
    
    # Verify training dataset creation
    train_call = MockConceptAwareDataset.call_args_list[0].kwargs
    assert train_call['root_dir'] == training_path
    assert train_call['concepts_file'] == mock_config.concepts_file
    assert train_call['transform'] == GTSRB_transform

    # Verify test dataset creation
    test_call = MockConceptAwareDataset.call_args_list[1].kwargs
    assert test_call['root_dir'] == test_path
    assert test_call['concepts_file'] == mock_config.concepts_file
    assert test_call['transform'] == GTSRB_transform
    
    # Assert that random_split was called with the correct arguments
    mock_random_split.assert_called_once()
    assert mock_random_split.call_args.args[0] is not None
    assert mock_random_split.call_args.args[1] == [80, 20]
    
    # Assert that the factory's instance variables are set
    assert factory.datasets_loaded is True
    assert len(factory.train_dataset) == 80
    assert len(factory.val_dataset) == 20
    assert len(factory.test_dataset) == 100
    
    # Assert that the method returns self
    assert result is factory

# Corrected test: We configure the return value of random_split
@patch('data_access.datasets.GTSRBFactory.ConceptAwareDataset', return_value=Mock(__len__=Mock(return_value=100)))
@patch('data_access.datasets.GTSRBFactory.random_split', return_value=[Mock(__len__=Mock(return_value=80)), Mock(__len__=Mock(return_value=20))])
def test_load_datasets_with_custom_transforms(mock_random_split, MockConceptAwareDataset, factory):
    """Test that load_datasets respects custom transforms."""
    custom_train_transform = transforms.ToTensor()
    custom_test_transform = transforms.Grayscale()

    factory.load_datasets(
        train_transform=custom_train_transform,
        test_transform=custom_test_transform
    )
    
    # Verify training dataset creation used the custom transform
    train_call = MockConceptAwareDataset.call_args_list[0].kwargs
    assert train_call['transform'] is custom_train_transform
    
    # Verify test dataset creation used the custom transform
    test_call = MockConceptAwareDataset.call_args_list[1].kwargs
    assert test_call['transform'] is custom_test_transform

# Corrected test: We configure the return value of random_split
@patch('data_access.datasets.GTSRBFactory.ConceptAwareDataset', return_value=Mock(spec=ConceptAwareDataset, __len__=Mock(return_value=100)))
@patch('data_access.datasets.GTSRBFactory.random_split', return_value=[Mock(__len__=Mock(return_value=80)), Mock(__len__=Mock(return_value=20))])
def test_load_datasets_sets_datasets_loaded_flag(mock_random_split, MockConceptAwareDataset, factory):
    """Test that the datasets_loaded flag is set after a successful load."""
    factory.load_datasets()
    assert factory.datasets_loaded is True
            
# Corrected test: We configure the return value of random_split
@patch('data_access.datasets.GTSRBFactory.ConceptAwareDataset', return_value=Mock(__len__=Mock(return_value=100)))
@patch('data_access.datasets.GTSRBFactory.random_split', return_value=[Mock(__len__=Mock(return_value=80)), Mock(__len__=Mock(return_value=20))])
def test_load_datasets_handles_no_concepts_file(mock_random_split, MockConceptAwareDataset, factory, mock_config):
    """Test the factory's behavior when concepts_file is None."""
    mock_config.concepts_file = None
    factory.load_datasets()

    train_call = MockConceptAwareDataset.call_args_list[0].kwargs
    assert train_call['concepts_file'] is None

    test_call = MockConceptAwareDataset.call_args_list[1].kwargs
    assert test_call['concepts_file'] is None