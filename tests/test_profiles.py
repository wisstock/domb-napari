import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy import stats

from domb_napari.core import profiles


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4000, (7, 20, 20)).astype(np.uint16)
    labels = np.zeros((20, 20), dtype=int)
    labels[1:5, 1:5] = 1
    labels[8:12, 3:9] = 2
    labels[15:19, 10:18] = 5
    return img, labels


def test_labels_to_profiles(data):
    img, labels = data
    prof = profiles.labels_to_profiles(labels, img)
    ids = profiles.label_ids(labels)
    assert list(ids) == [1, 2, 5]
    assert prof.shape == (3, 7) and prof.dtype == np.float32
    ref = np.array([ndi.mean(frame, labels, ids) for frame in img]).T
    np.testing.assert_allclose(prof, ref, rtol=1e-6)


def test_labels_without_background(data):
    img, _ = data
    labels = np.ones((20, 20), dtype=int)
    labels[:, 10:] = 2
    prof = profiles.labels_to_profiles(labels, img)
    assert prof.shape[0] == 2
    np.testing.assert_allclose(prof[0], img[:, :, :10].mean(axis=(1, 2)), rtol=1e-6)


def test_input_is_not_modified(data):
    img, labels = data
    img_copy = img.copy()
    profiles.labels_to_profiles(labels, img)
    assert np.array_equal(img, img_copy)


def test_baselines():
    prof = np.array([[10.0, 10.0, 20.0, 30.0], [0.0, 0.0, 5.0, 5.0]])
    np.testing.assert_allclose(profiles.delta_prof_simple(prof, win_size=2, mode='dF'),
                               [[0, 0, 10, 20], [0, 0, 5, 5]])
    rel = profiles.delta_prof_simple(prof, win_size=2, mode='dF/F0')
    np.testing.assert_allclose(rel[0], [0, 0, 1, 2])
    assert np.all(np.isnan(rel[1]))
    np.testing.assert_allclose(profiles.delta_prof_simple(prof, win_size=2, mode='abs'), prof)


def test_profile_stats():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (12, 50))     # 12 ROIs, 50 frames
    center, sem = profiles.profile_stats(x, 'se')
    np.testing.assert_allclose(center, x.mean(axis=0))
    np.testing.assert_allclose(sem, stats.sem(x, axis=0))
    _, ci = profiles.profile_stats(x, 'ci')
    np.testing.assert_allclose(ci, stats.t.ppf(0.975, 11) * stats.sem(x, axis=0))
    med, iqr = profiles.profile_stats(x, 'iqr')
    assert iqr.shape == (2, 50)
    np.testing.assert_allclose(iqr.sum(axis=0), stats.iqr(x, axis=0))
    np.testing.assert_allclose(med, np.median(x, axis=0))


def test_profiles_table(data):
    img, labels = data
    prof = profiles.labels_to_profiles(labels, img)
    ids = profiles.label_ids(labels)
    dist = profiles.stim_distances(labels, (0, 0))
    df = profiles.profiles_table(prof, ids, time_scale=0.5, img_name='img', labels_name='lab',
                                 distances=dist)
    assert list(df.columns) == ['id', 'lab_id', 'roi', 'dist', 'index', 'time',
                                'abs_int', 'dF_int', 'dF/F0_int', 'base']
    assert len(df) == 2 * 3 * 7
    assert sorted(df['roi'].unique()) == [1, 2, 5]
    assert set(df['base']) == {'simple', 'dietrich'}
    first = df[(df['roi'] == 5) & (df['base'] == 'simple')]
    assert list(first['index']) == list(range(7))
    np.testing.assert_allclose(first['time'], np.arange(7) * 0.5)
    np.testing.assert_allclose(first['abs_int'], np.round(prof[2].astype(float), 4))
    assert dist[0] < dist[1] < dist[2]
    assert 'dist' not in profiles.profiles_table(prof, ids).columns
