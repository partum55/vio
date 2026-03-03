# vio

VIO viewer with trajectory + point-cloud visualization and video recording.

## Python implementation

The new Python implementation lives under `python/vio_py` and provides feature parity for:
- synthetic demo generation
- loading TUM trajectory files
- loading XYZ point clouds
- interactive viewing
- MP4 recording via `ffmpeg`

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

`ffmpeg` must be available on your `PATH`.
For Matplotlib browser fallback (`--backend mpl` in headless/Wayland problem cases), `tornado` is required and included in `python/requirements.txt`.

### Usage

Run from the repository root:

```bash
PYTHONPATH=python python3 -m vio_py.cli --help
PYTHONPATH=python python3 -m vio_py.cli
PYTHONPATH=python python3 -m vio_py.cli --backend auto
PYTHONPATH=python python3 -m vio_py.cli --backend mpl
PYTHONPATH=python python3 -m vio_py.cli --trajectory traj.tum
PYTHONPATH=python python3 -m vio_py.cli --cloud points.xyz
PYTHONPATH=python python3 -m vio_py.cli --trajectory t.tum --cloud c.xyz
PYTHONPATH=python python3 -m vio_py.cli --record out.mp4 --trajectory t.tum --backend auto
PYTHONPATH=python python3 -m vio_py.cli --record out.mp4 --backend open3d
```

From the `python/` directory, both forms also work:

```bash
python3 -m vio_py --help
python3 vio_py --help
```

### Backend Selection

Use `--backend` to control rendering:
- `auto` (default): try Open3D first, then fallback to Matplotlib when Open3D fails.
- `open3d`: require Open3D; fails if a GL context cannot be created.
- `mpl`: force Matplotlib backend.

### Wayland/Open3D Troubleshooting

If Open3D shows GLFW/GLEW context errors on Wayland, run:

```bash
python3 vio_py --backend mpl
python3 vio_py --record out.mp4 --backend mpl
```

If you need native Open3D interactive rendering, run under X11/XWayland.

## C++ implementation

The original C++/Pangolin implementation remains available.

```bash
cmake -S . -B build
cmake --build build -j
./build/vio_viewer --help
```

## Team

[Nazar Mykhailyshchuk](https://github.com/partum55), [Marharyta Paduchak](https://github.com/marharytapaduchak), [Alina Bodnar](https://github.com/alinabodnarpn), [Daryna Shevchuk](https://github.com/dasha-pn)
