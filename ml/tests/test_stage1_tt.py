from pathlib import Path

import pytest

from train.stage1_tt import expand_classifier_moments, external_rows


def test_public_test_namespace_and_row_order(tmp_path):
    (tmp_path / 'train_list_start0.txt').write_text('00002/first.jpg 0 1\n00002/second.jpg 0 2\n')
    (tmp_path / 'test_10000_id_query.txt').write_text('00001/z.jpg 1 3\n')
    (tmp_path / 'test_10000_id.txt').write_text('00001/a.jpg 1 4\n')
    rows, ids = external_rows(tmp_path, tmp_path / 'train', tmp_path / 'test')
    assert ids == ['veriwild:00002', 'veriwild_test:1']
    assert [r['label'] for r in rows] == [0, 0, 1, 1]
    assert [r['camera_id'] for r in rows] == ['1', '2', '4', '3']
    assert [Path(r['path']).name for r in rows[-2:]] == ['a.jpg', 'z.jpg']


def test_external_metadata_conflict_is_rejected(tmp_path):
    (tmp_path / 'train_list_start0.txt').write_text('00002/a.jpg 0 1\n')
    (tmp_path / 'test_10000_id_query.txt').write_text('00001/a.jpg 1 3\n')
    (tmp_path / 'test_10000_id.txt').write_text('00001/a.jpg 1 4\n')
    with pytest.raises(ValueError, match='Conflicting'):
        external_rows(tmp_path, tmp_path, tmp_path)


def test_classifier_adamw_moments_follow_identity_mapping():
    torch = pytest.importorskip('torch')
    values = torch.tensor([[1., 2.], [3., 4.]])
    state = {'param_groups': [{'params': [0]}, {'params': [1, 2]}],
             'state': {0: {'exp_avg': torch.tensor([9.])},
                       2: {'exp_avg': values.clone(), 'exp_avg_sq': values.square(), 'step': torch.tensor(34726.)}}}
    result = expand_classifier_moments(state, [2, 0], [0, 1], 3)
    assert torch.equal(result['state'][2]['exp_avg'], torch.tensor([[3., 4.], [0., 0.], [1., 2.]]))
    assert torch.equal(result['state'][2]['exp_avg_sq'], torch.tensor([[9., 16.], [0., 0.], [1., 4.]]))
    assert result['state'][2]['step'].item() == 34726
    assert result['state'][0]['exp_avg'].item() == 9
