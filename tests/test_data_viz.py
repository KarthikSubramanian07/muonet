"""Data generation, web export, and cross-section rendering."""

import json

import numpy as np

from muonet.data import collect_measurement, make_pair, poca_input
from muonet.detector import get_detector_config
from muonet.geometry import DetectorGeometry, Primitive, Scene, VoxelGrid
from muonet.seeding import make_rng
from muonet.viz import export_volume, save_cross_sections


def _small_scene():
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 0), (10, 10, 10))]
    )
    return scene, grid, DetectorGeometry(object_half_z_cm=16.0)


def test_collect_measurement_exact_count():
    scene, _, det = _small_scene()
    cfg = get_detector_config("realistic_sipm")
    meas = collect_measurement(scene, det, cfg, 300, make_rng(0))
    assert len(meas) == 300
    assert meas.valid.all()


def test_poca_input_normalized():
    scene, grid, det = _small_scene()
    cfg = get_detector_config("ideal")
    meas = collect_measurement(scene, det, cfg, 400, make_rng(0))
    vol = poca_input(meas, grid)
    assert vol.shape == grid.shape
    assert vol.min() >= 0.0 and vol.max() <= 1.0


def test_make_pair_shapes():
    scene, grid, det = _small_scene()
    cfg = get_detector_config("ideal")
    x, y, hz = make_pair(scene, det, cfg, 300, make_rng(0))
    assert x.shape == y.shape == grid.shape
    assert hz.dtype == bool


def test_export_volume_writes_bin_and_header(tmp_path):
    vol = np.zeros((8, 8, 8))
    vol[2:5, 2:5, 2:5] = 1.0
    header = export_volume(vol, str(tmp_path / "v"))
    assert header["shape"] == [8, 8, 8]
    bin_bytes = (tmp_path / "v.bin").read_bytes()
    assert len(bin_bytes) == 8 * 8 * 8  # one uint8 per voxel
    parsed = json.loads((tmp_path / "v.json").read_text())
    assert parsed["dtype"] == "uint8"
    assert parsed["layout"] == "x-fastest"


def test_export_volume_x_fastest_layout(tmp_path):
    # A single lit voxel at (x=1,y=0,z=0) must land at linear index 1.
    vol = np.zeros((4, 4, 4))
    vol[1, 0, 0] = 1.0
    export_volume(vol, str(tmp_path / "v"), normalize=False)
    data = np.frombuffer((tmp_path / "v.bin").read_bytes(), dtype=np.uint8)
    assert data[1] == 255
    assert data[0] == 0


def test_save_cross_sections(tmp_path):
    vol = np.zeros((16, 16, 16))
    vol[6:10, 6:10, 6:10] = 1.0
    out = save_cross_sections(vol, str(tmp_path / "slice.png"), title="test")
    assert (tmp_path / "slice.png").exists()
    assert out.endswith(".png")
