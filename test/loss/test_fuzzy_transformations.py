import torch
import pytest
from torch import nn
from models.loss.fuzzy_transformations import GodelTNorm, GodelTConorm, GodelAAggregation, GodelEAggregation
from models.loss.custom_fuzzy_loss import ExactlyOneShape

# the relevant concept indices are the first 3
test_batch = torch.tensor([[0,0,0,1],[1,0,0,1],[1,1,1,0],[0,1,1,0]])
test_tensor = torch.tensor([0,1,0])
expected_loss_batch = torch.tensor([1,0,1,1])
expected_loss_tensor = torch.tensor([0])

def test_godel_Aaggregation_batch():
    godel_agg = GodelAAggregation()
    assert torch.equal(torch.tensor([0,0,1,0]), godel_agg(test_batch))

def test_godel_aggregation_1d_tensor():
    godel_agg = GodelAAggregation()
    result = godel_agg(test_tensor)
    assert torch.equal(torch.tensor(0), result)

def test_godel_Eaggregation_batch():
    godel_agg = GodelEAggregation()
    assert torch.equal(torch.tensor([0,1,1,1]), godel_agg(test_batch))

def test_godel_Eaggregation_1d_tensor():
    godel_agg = GodelEAggregation()
    result = godel_agg(test_tensor)
    assert torch.equal(torch.tensor(1), result)

def test_godel_tnorm():
    a = torch.tensor([0,1,0,1])
    b = torch.tensor([0,0,1,1])
    godel_tnorm = GodelTNorm()
    assert torch.equal(torch.tensor([0,0,0,1]), godel_tnorm(a,b))
    assert torch.equal(torch.tensor(0), godel_tnorm(torch.tensor(0),torch.tensor(1)))

def test_godel_tconorm():
    a = torch.tensor([0,1,0,1])
    b = torch.tensor([0,0,1,1])
    godel_tconorm = GodelTConorm()
    assert torch.equal(torch.tensor([0,1,1,1]), godel_tconorm(a,b))
    assert torch.equal(torch.tensor(1), godel_tconorm(torch.tensor(0),torch.tensor(1)))

def test_exactly_one_shape_godel():
    godel_t_norm = GodelTNorm()
    godel_t_conorm =  GodelTConorm()
    godel_e_agg = GodelEAggregation()
    godel_a_agg = GodelAAggregation()
    loss = ExactlyOneShape(t_norm=godel_t_norm, t_conorm=godel_t_conorm, implication=None, e_aggregation=godel_e_agg, a_aggregation=godel_a_agg, shape_indices=[0,1,2])
    assert torch.equal(loss(test_batch), expected_loss_batch)

def test_exactly_one_shape_godel_vector():
    godel_t_norm = GodelTNorm()
    godel_t_conorm =  GodelTConorm()
    godel_e_agg = GodelEAggregation()
    godel_a_agg = GodelAAggregation()
    loss = ExactlyOneShape(t_norm=godel_t_norm, t_conorm=godel_t_conorm, implication=None, e_aggregation=godel_e_agg, a_aggregation=godel_a_agg, shape_indices=[0,1,2])
    assert torch.equal(loss(test_tensor), expected_loss_tensor)
