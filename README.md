# nam.cpp

![](https://repository-images.githubusercontent.com/1298156726/b39879cc-f47d-433e-907f-94f1b2bbb84c)

An experimental Neural Amp Modeler (NAM) inference implementation using [ggml](https://github.com/ggml-org/ggml).

It currently supports classic A1 WaveNet models and offline processing of mono float32 WAV files only.

## Build

Requires `json-c`, `pkg-config`, and a local ggml source checkout.

```sh
cmake -B build -DGGML_DIR=/path/to/ggml
cmake --build build -j
```

## Usage

```sh
./build/nam [-t N] [-b N] model.nam input.wav output.wav
```

By default, the complete file is processed in one offline graph. Passing a positive `-b`/`--block` size uses a reusable fixed-size graph and carries the model's input history between blocks. This bounds inference memory while producing the same causal output; for example:

```sh
./build/nam --block 65536 model.nam input.wav output.wav
```
