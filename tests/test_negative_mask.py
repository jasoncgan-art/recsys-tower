import torch
from train import inbatch_neg_mask


def test_duplicate_positive_columns_are_all_masked_except_diagonal():
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([5, 5, 7])
    positives = torch.tensor([[5, -1], [5, -1], [5, 7]])
    actual = inbatch_neg_mask(users, items, positives, 8)
    expected = torch.tensor([[False, True, False], [True, False, False],
                             [True, True, False]])
    assert torch.equal(actual, expected)


def test_padding_is_not_item_zero_and_real_negatives_survive():
    users = torch.tensor([0, 1])
    items = torch.tensor([0, 2])
    positives = torch.tensor([[0, -1], [2, -1]])
    assert not inbatch_neg_mask(users, items, positives, 3).any()
