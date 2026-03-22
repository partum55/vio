# vio

VIO trajectory replay with rerun-based real-time visualization.

## Python rerun receiver

The Python side is a small TCP receiver that logs incoming pose and IMU samples to rerun.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

### Usage

Run from the repository root:

```bash
PYTHONPATH=python python3 -m vio_py.cli --help
PYTHONPATH=python python3 -m vio_py.cli --host 127.0.0.1 --port 9876
```

This starts the rerun logger:

- `rr.init("vio")`
- `rr.spawn()`
- TCP receive loop on the requested host/port

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

Run the receiver first, then stream samples from C++:

```bash
PYTHONPATH=python python3 -m vio_py.cli --host 127.0.0.1 --port 9876
./build/vio_viewer --host 127.0.0.1 --port 9876
./build/vio_viewer --trajectory traj.tum --host 127.0.0.1 --port 9876
```

The C++ side sends newline-delimited JSON samples containing:

- pose translation
- pose quaternion in `xyzw`
- derived accelerometer sample
- derived gyroscope sample

Point cloud input is still accepted for compatibility but is not visualized by the rerun pipeline.

## Team

[Nazar Mykhailyshchuk](https://github.com/partum55), [Marharyta Paduchak](https://github.com/marharytapaduchak), [Alina Bodnar](https://github.com/alinabodnarpn), [Daryna Shevchuk](https://github.com/dasha-pn)
