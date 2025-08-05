import pytest
import torch
from torch.utils.data import Dataset, DataLoader

# Import the class you want to test
from data_access.data_factory import DatasetFactory, seed_worker 

# ---- Mocks and Fixtures for Testing ----

class MockConfig:
    """A mock configuration object for testing."""
    def __init__(self, **kwargs):
        self.batch_size = 4
        self.shuffle_dataset = True
        self.num_workers = 0
        self.pin_memory = False
        self.seed = 42
        self.val_split = 0.2
        self.data_path = 'mock_data'
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockDataset(Dataset):
    """A mock dataset with a fixed length and deterministic data."""
    def __init__(self, length=100):
        self._length = length
    def __len__(self):
        return self._length
    def __getitem__(self, idx):
        # Return a deterministic tensor based on the index.
        # This ensures that calling __getitem__ with the same idx
        # always returns the same tensor.
        image_tensor = torch.full((3, 32, 32), float(idx))
        label_tensor = torch.tensor([idx % 10])
        return image_tensor, label_tensor

class MockConcreteFactory(DatasetFactory):
    """A concrete implementation of the abstract factory for testing."""
    def __init__(self, config):
        super().__init__(config)
        self.load_datasets_called = False

    def load_datasets(self):
        """Simulates loading datasets and setting the instance variables."""
        self.train_dataset = MockDataset(length=80)
        self.val_dataset = MockDataset(length=20)
        self.test_dataset = MockDataset(length=10)
        self.datasets_loaded = True
        self.load_datasets_called = True

@pytest.fixture
def mock_config():
    """Fixture to provide a standard mock config object."""
    return MockConfig()

@pytest.fixture
def mock_factory(mock_config):
    """Fixture to provide a fresh MockConcreteFactory instance for each test."""
    return MockConcreteFactory(mock_config)

# ---- Pytest Tests ----

def test_initial_state(mock_factory):
    """Test that the factory is initialized with the correct state."""
    assert not mock_factory.datasets_loaded
    assert not mock_factory.dataloaders_set

    assert mock_factory.train_dataset is None
    assert mock_factory.val_dataset is None
    assert mock_factory.test_dataset is None

    assert mock_factory.train_dataloader is None
    assert mock_factory.val_dataloader is None
    assert mock_factory.test_dataloader is None

def test_get_datasets_lazy_loading(mock_factory, mocker):
    """Test that get_datasets calls load_datasets only once and returns the datasets."""
    mocker.patch.object(mock_factory, 'load_datasets', wraps=mock_factory.load_datasets)
    
    # First call should load the datasets
    train_ds, val_ds, test_ds = mock_factory.get_datasets()
    
    assert mock_factory.datasets_loaded
    mock_factory.load_datasets.assert_called_once()
    assert isinstance(train_ds, MockDataset)
    assert len(train_ds) == 80
    assert len(val_ds) == 20
    assert len(test_ds) == 10
    
    # Subsequent calls should not call load_datasets again
    mock_factory.load_datasets.reset_mock()
    mock_factory.get_datasets()
    mock_factory.load_datasets.assert_not_called()
    
def test_get_datasets_no_load(mock_factory):
    """Test that get_datasets returns (None, None, None) when load_if_none is False."""
    train_ds, val_ds, test_ds = mock_factory.get_datasets(load_if_none=False)
    
    assert not mock_factory.datasets_loaded
    assert train_ds is None
    assert val_ds is None
    assert test_ds is None

def test_wrap_dataloader(mock_factory):
    """Test that the private _wrap_dataloader method returns a valid DataLoader."""
    dataset = MockDataset()
    dataloader = mock_factory._wrap_dataloader(dataset)

    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == mock_factory.config.batch_size
    assert dataloader.num_workers == mock_factory.config.num_workers

def test_wrap_dataloader_raises_error(mock_factory):
    """Test that _wrap_dataloader raises a ValueError for a None dataset."""
    with pytest.raises(ValueError, match="Dataset is None"):
        mock_factory._wrap_dataloader(None)

def test_set_dataloaders_lazy_loading(mock_factory, mocker):
    """Test that set_dataloaders correctly calls load_datasets if needed."""
    mocker.patch.object(mock_factory, 'load_datasets', wraps=mock_factory.load_datasets)
    
    assert not mock_factory.datasets_loaded
    mock_factory.set_dataloaders()
    
    assert mock_factory.datasets_loaded
    assert mock_factory.dataloaders_set
    assert isinstance(mock_factory.train_dataloader, DataLoader)
    assert isinstance(mock_factory.val_dataloader, DataLoader)
    assert isinstance(mock_factory.test_dataloader, DataLoader)
    mock_factory.load_datasets.assert_called_once()
    
def test_set_dataloaders_no_load(mock_factory):
    """Test that set_dataloaders returns self without loading if load_if_none is False."""
    assert not mock_factory.datasets_loaded
    result = mock_factory.set_dataloaders(load_if_none=False)
    
    assert not mock_factory.datasets_loaded
    assert not mock_factory.dataloaders_set
    assert result is mock_factory

def test_get_dataloaders_lazy_loading(mock_factory, mocker):
    """Test that get_dataloaders calls set_dataloaders if needed."""
    mocker.patch.object(mock_factory, 'set_dataloaders', wraps=mock_factory.set_dataloaders)
    
    assert not mock_factory.dataloaders_set
    train_dl, _, _ = mock_factory.get_dataloaders()

    mock_factory.set_dataloaders.assert_called_once()
    assert mock_factory.datasets_loaded
    assert mock_factory.dataloaders_set
    assert isinstance(train_dl, DataLoader)
    
    # Subsequent calls should not re-create dataloaders
    mock_factory.set_dataloaders.reset_mock()
    mock_factory.get_dataloaders()
    mock_factory.set_dataloaders.assert_not_called()

def test_get_dataloaders_no_load(mock_factory):
    """Test that get_dataloaders returns (None, None, None) when load_if_none is False."""
    assert not mock_factory.dataloaders_set
    train_dl, val_dl, test_dl = mock_factory.get_dataloaders(load_if_none=False)

    assert not mock_factory.datasets_loaded
    assert not mock_factory.dataloaders_set
    assert train_dl is None
    assert val_dl is None
    assert test_dl is None
    
def test_config_attributes_with_concepts():
    """Test that the factory correctly handles config with concepts file."""
    mock_config_with_concepts = MockConfig(
        concepts_file='concepts.csv'
    )
    factory_with_concepts = MockConcreteFactory(mock_config_with_concepts)
    assert factory_with_concepts.concepts_file == 'concepts.csv'

def test_config_attributes_without_concepts():
    """Test that the factory defaults to None/False without concepts file in config."""
    mock_config_without_concepts = MockConfig()
    factory_without_concepts = MockConcreteFactory(mock_config_without_concepts)
    assert factory_without_concepts.concepts_file is None

def test_seeding_reproducibility(mock_config):
    """Test that the dataloaders are seeded for reproducibility."""
    factory1 = MockConcreteFactory(mock_config)
    train_dl1, _, _ = factory1.get_dataloaders()
    
    factory2 = MockConcreteFactory(mock_config)
    train_dl2, _, _ = factory2.get_dataloaders()
    
    batch1 = next(iter(train_dl1))
    batch2 = next(iter(train_dl2))
    
    assert torch.equal(batch1[0], batch2[0])
    assert torch.equal(batch1[1], batch2[1])