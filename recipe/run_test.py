import tempfile
from pathlib import Path

import numpy as np
import xatlas


def _assert_atlas_result(vmapping, indices, uvs, vertex_count):
    assert vmapping.dtype == np.uint32
    assert indices.dtype == np.uint32
    assert uvs.dtype == np.float32
    assert indices.ndim == 2
    assert indices.shape[1] == 3
    assert uvs.shape == (len(vmapping), 2)
    assert np.all(vmapping < vertex_count)
    assert np.all(indices < len(vmapping))
    assert np.all(np.isfinite(uvs))


def main():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    _assert_atlas_result(vmapping, indices, uvs, len(vertices))

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    atlas.generate()
    assert atlas.mesh_count == 1
    assert len(atlas) == 1
    atlas_mapping, atlas_indices, atlas_uvs = atlas[0]
    _assert_atlas_result(atlas_mapping, atlas_indices, atlas_uvs, len(vertices))

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = Path(tmpdir) / "atlas.obj"
        xatlas.export(str(obj_path), vertices[vmapping], indices, uvs)
        obj_text = obj_path.read_text()
        assert "\nv " in "\n" + obj_text
        assert "\nvt " in "\n" + obj_text
        assert "\nf " in "\n" + obj_text


if __name__ == "__main__":
    main()
