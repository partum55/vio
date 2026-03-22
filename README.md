# vio

Rerun-based visualization branch for trajectories and point clouds.

This branch contains only the visualization stack:
- C++ producer for synthetic scenes or TUM/XYZ inputs
- Python Rerun receiver for live 3D visualization

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

## Python receiver

Run from the repository root:

```bash
PYTHONPATH=python python3 -m vio_py.cli --help
PYTHONPATH=python python3 -m vio_py.cli --host 127.0.0.1 --port 9877
```

From the `python/` directory, both forms also work:

```bash
python3 -m vio_py --help
python3 vio_py --help
```

## C++ producer

```bash
cmake -S . -B build
cmake --build build -j
./build/vio_viewer --help
```

Synthetic demo:

```bash
./build/vio_viewer
```

Load a trajectory and point cloud:

```bash
./build/vio_viewer --trajectory traj.tum --cloud points.xyz
```

Connect to an already running receiver:

```bash
./build/vio_viewer --host 127.0.0.1 --port 9877 --no-receiver
```

## Team

[Nazar Mykhailyshchuk](https://github.com/partum55), [Marharyta Paduchak](https://github.com/marharytapaduchak), [Alina Bodnar](https://github.com/alinabodnarpn), [Daryna Shevchuk](https://github.com/dasha-pn)
