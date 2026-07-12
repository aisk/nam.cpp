# nam.cpp

An experimental NAM inference implementation using ggml.

It currently supports classic A1 WaveNet models and offline processing of mono float32 WAV files only.

## Build

Requires `json-c`, `pkg-config`, and a local ggml source checkout.

```sh
cmake -B build -DGGML_DIR=/path/to/ggml
cmake --build build -j
```

## Usage

```sh
./build/nam [--threads N] model.nam input.wav output.wav
```
