# Qrack

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5812507.svg)](https://doi.org/10.5281/zenodo.5812507) [![Mentioned in Awesome awesome-quantum-computing](https://awesome.re/mentioned-badge.svg)](https://github.com/desireevl/awesome-quantum-computing)

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

The open source vm6502q/qrack library and its associated plugins and projects under the vm6502q organization header comprise a framework for full-stack quantum computing development, via high performance and fundamentally optimized simulation. The intent of "Qrack" is to provide maximum performance for the simulation of an ideal, virtually error-free quantum computer, across the broadest possible set of hardware and operating systems.

Using the C++11 standard, at base, Qrack has an external-dependency-free CPU simulator "engine," as well as a GPU simulator engine that depends only on OpenCL. The `QUnit` layer provides novel, fundamental optimizations in the simulation algorithm, based on "[Schmidt decomposition](https://arxiv.org/abs/1710.05867)," transformation of basis, 2 qubit controlled gate buffer caching, the physical nonobservability of arbitrary global phase factors on a state vector, and many other "synergistic" and incidental points of optimization between these approaches and in addition to them. `QUnit` can be placed "on top" of either CPU, GPU, or hybrid engine types, and an additional `QPager` layer can sit between these, or in place of `QUnit`. Optimizations and hardware support are highly configurable, particularly at build time.

A `QUnit` or `QEngine` can be thought of as like simply a one-dimensional array of qubits, within which any qubit has the capacity to directly and fully entangle with any and all others. Bits can be manipulated on by a single bit gate at a time, or gates and higher level quantum instructions can be acted over arbitrary contiguous sets of bits. A qubit start index and a length is specified for parallel operation of gates over bits or for higher level instructions, like arithmetic on abitrary width registers. Some methods are designed for (bitwise and register-like) interface between quantum and classical bits. See the Doxygen for the purpose of gate-like and register-like functions.

Qrack has already been integrated with a [MOS 6502 emulator](https://github.com/vm6502q/vm6502q), which demonstrates Qrack's original purpose, for use in developing chip-like quantum computer emulators. (The base 6502 emulator to which Qrack was added for that project is by Marek Karcz, many thanks to Marek! See https://github.com/makarcz/vm6502.)

A number of useful "pseudo-quantum" operations, which could not be carried out by true hardware quantum computers easily or at all, are included in the API for purposes like debugging, but also for potential speed-up, by allowing the classical emulation to leverage nonphysical exceptions to quantum logic, like by cloning a quantum state. In practice, though, quantum circuit programs might not rely on this (nonphysical) functionality at all. (The MOS 6502 emulator referenced above does not. It happens, many of these methods will not be exotic at all, to those familiar with other major quantum computing libraries. For example, notably, simply returning a full vector of probability amplitudes actually qualifies as "pseudo-quantum," in this sense.)

Qrack compiles like a library. To include in your project:

1. In your source code:

```cpp
#include "qrack/qfactory.hpp"
```

2. On the command line, in the project directory

```sh
$ mkdir _build && cd _build && cmake .. && make all install
```

Instantiate a `Qrack::QUnit`, specifying the desired number of qubits. (Optionally, also specify the initial bit permutation state in the constructor.) `QUnit`s can be (Schmidt) "composed" and "decomposed" with and from each other, to join and separate the representations of qubit "registers" that are not entangled at the point (de)composition. Both single quantum gate commands and register-like multi-bit commands are available.

For distributed simulation, the `Qrack::QPager` layer will segment a single register into a power-of-two count of equal length pages, running on an arbitrary number of OpenCL accelerators. The `QPager` layer also scales to arbitrarily small as well as large qubit counts, such that it can be appropriate for use on a single accelerator for small width simulations. The `QPager` layer is also compatible with Clifford set preamble circuits simulated with `QStabilizerHybrid`, as a layer over `QPager`, and `QHybrid` for CPU/GPU switching can be used as the "engine" layer under it. For Qrack in a cluster environment, we support the SnuCL and VirtualCL OpenCL virtualization layers, with OpenCL v1.1 compliant host code without required "host pointers."

For more information, compile the `doxygen.config` in the root folder, and then check the `doc` folder.

## PyQrack Source Build

The CMake settings for default build of [PyQrack](https://github.com/unitaryfund/pyqrack) are as follows, (assuming you are in a build directory created inside the top-level directory of the repo clone):

x86-64 Linux (OpenCL):
```
cmake -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DQBCAPPOW=12 -DCPP_STD=14 -DUINTPOW=5 ..
```

x86-64 Linux (CUDA):
```
cmake -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DQBCAPPOW=12 -DCPP_STD=14 -DENABLE_OPENCL=OFF -DENABLE_CUDA=ON -DUINTPOW=5 ..
```

x86-64 Mac (might need `-Werror`, "warning to error," disabled in CMake files):
```
cmake -DQBCAPPOW=12 -DUINTPOW=5 -DCPP_STD=14 ..
```

RISC (ARM) Linux:
```
cmake -DENABLE_RDRAND=OFF -DENABLE_DEVRAND=ON -DENABLE_COMPLEX_X2=OFF -DQBCAPPOW=12 -DUINTPOW=5 -DCPP_STD=14 ..
```

[Emscripten (WASM)](https://qrack.net/):
```
emcmake cmake -DENABLE_RDRAND=OFF -DUINTPOW=5 -DENABLE_PTHREAD=OFF -DSEED_DEVRAND=OFF -DQBCAPPOW=12 -DUINTPOW=5 -DCPP_STD=14 ..
```

Windows-based systems are more specific, but there is a bit more information about them further below.

## Documentation

Live version of the documentation, including API reference, can be obtained at: https://qrack.readthedocs.io/en/latest/

## Community

Qrack has a community home at the Advanced Computing Topics server on Discord, at: https://discordapp.com/invite/Gj3CHDy

For help getting started with contributing, see our [CONTRIBUTING.md](CONTRIBUTING.md).

## Installing Qrack

If you're on Ubuntu 18.04, 20.04, or 22.04 LTS, you're in luck: Qrack manages a PPA that provides binary installers for _all_ available CPU architectures (except any that require administrative attention from Ubuntu or Canonical).

```sh
    $ sudo add-apt-repository ppa:wrathfulspatula/vm6502q
    $ sudo apt update
    $ sudo apt install libqrack-dev
```

(You might need to install the `add-apt-repository` tool first, through `apt` itself.)

Otherwise, standardized builds are available on the [releases](https://github.com/unitaryfund/qrack/releases) page.

If you're looking for [PyQrack](https://github.com/unitaryfund/pyqrack), know that the PyPi package has a self-contained Qrack release. (On Ubuntu, the PyPi package can be used, but it is **strongly recommended** that you instead install the `libqrack` or `libqrack-dev` packages from the PPA, as above, then install `main` branch PyQrack from source, which will use the Ubuntu `apt` packages.)

## test/tests.cpp

The included `test/tests.cpp` contains unit tests and usage examples. The unittests themselves can be executed:

```sh
    $ _build/unittest
```

Similarly, benchmarks are in `test/benchmarks.cpp`:

```sh
    $ _build/benchmarks [--optimal] [--max-qubits=30] [test_qft_cosmology]
```

## OpenCL on systems prior to OpenCL v2.0

Particularly on older hardware, it is possible that you do not have OpenCL v2.0 available. In theory, Qrack should work off-the-shelf anyway. However, if the OpenCL implementation isn't even aware of the existence of v2.0, use the following option to completely manually force all v2.0 functionality off and to set the target OpenCL API level expressly to target v1.2 and minimum level v1.1:

```sh
    $ cmake -DENALBE_OOO_OCL=OFF ..
```

## Installing OpenCL on VMWare

Most platforms offer a standardized way of installing OpenCL. However, a method for VMWare benefits from documentation, here.

1.  Download the [AMD APP SDK](https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)
1.  Install it.
1.  Add symlinks for `/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/libOpenCL.so.1` to `/usr/lib`
1.  Add symlinks for `/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/libamdocl64.so` to `/usr/lib`
1.  Make sure `clinfo` reports back that there is a valid backend to use (anything other than an error should be fine).
1.  Install OpenGL headers: `$ sudo apt install mesa-common-dev`
1.  Adjust the `makefile` to have the appropriate search paths

## Installing OpenCL on Mac

While the OpenCL framework is available by default on most modern Macs, the C++ header `cl.hpp` is usually not. One option for building for OpenCL on Mac is to download this header file and include it in the Qrack project folder under include/OpenCL (as `cl.hpp`). The OpenCL C++ header can be found at the Khronos OpenCL registry:

https://www.khronos.org/registry/OpenCL/

Otherwise, Homebrew offers a package with the headers: [opencl-clhpp-headers](https://formulae.brew.sh/formula/opencl-clhpp-headers) is the preferred method of installing headers, if `brew` is available.

## Building and Installing Qrack on Windows

Qrack supports building on Windows, but some special configuration is required. Windows 10 usually comes with default OpenCL libraries for Intel (or AMD) CPUs and their graphics coprocessors, but NVIDIA graphics card support might require the CUDA Toolkit. The CUDA Toolkit also provides an OpenCL development environment, which is generally necessary to build Qrack.

Qrack requires the `xxd` command to convert its OpenCL kernel code into hexadecimal format for building. `xxd` is not natively available on Windows systems, but Windows executables for it are provided by sources including the [Vim editor Windows port](https://www.vim.org/download.php).

CMake on Windows will set up a 32-bit Visual Studio project by default, (if using Visual Studio,) whereas 64-bit will probably be typically desired. `-DFPPOW=6` is used to set the systemic floating point accuracy to `double`, which is typically necessary for Q# accuracy tolerances. Putting together all of the above considerations, after installing the CUDA Toolkit and Vim, a typical CMake command for Windows might look like this:

```sh
    $ mkdir _build
    $ cd _build
    $ cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DXXD_BIN="C:/Program Files (x86)/Vim/vim82/xxd.exe" -DFPPOW=6 ..
```

After CMake, the project must be built in Visual Studio. Once installed, the `qrack_pinvoke` DLL is compatible with the Qrack Q# runtime fork, to provide `QrackSimulator`.

## C++ language standard

To change the C++ language standard language with which Qrack is applied, use `-DCPP_STD=n`, where "`n`" is the two-digit standard year:

```sh
cmake -DCPP_SIM=14 ..
```

By default, Qrack builds for C++11. For minimum support for all optional dependencies, C++14 or later might be required.

## Optional CUDA instead of OpenCL

Theoretically, building with CUDA for your native supported architectures is as simple as installing the CUDA toolkit and compiler and using this CMake command:

```sh
cmake -DENABLE_CUDA=ON [-DENABLE_OPENCL=OFF] [-DQRACK_CUDA_ARCHITECTURES=86] ..
```

where `-DENABLE_CUDA=ON` is required to enable CUDA, `-DENABLE_OPENCL=OFF` will cause CUDA to be used in the default optimal simulation layer stack instead of OpenCL, and `-DQRACK_CUDA_ARCHITECTURES` optionally specifies an explicit list of CUDA architectures for which to build. (If `-DQRACK_CUDA_ARCHITECTURES` is not set, Qrack will attempt to detect your native installed GPU architectures and build for exactly that set.)

## WebAssembly (WASM) builds

By nature of its pure C++11 design, Qrack happens to offer excellent compatibility with Emscripten ("WebAssembly") projects. See [the qrack.net repository](https://github.com/vm6502q/qrack.net) for an example and [qrack.net](https://qrack.net) for a live demo. OpenCL GPU operation is not yet available for WASM builds. While CPU multithreading might be possible in WASM, it is advisable that `pthread` usage and linking is disabled for most conventional Web applications, with `-DENABLE_PTHREAD=OFF` in CMake:

```sh
emcmake cmake -DENABLE_RDRAND=OFF -DENABLE_PTHREAD=OFF -DSEED_DEVRAND=OFF -DUINTPOW=5 ..
```

`-DUINTPOW=5` is optional, but WASM RAM limitations currently preclude >=32 qubits of potentially entangled state vector, so 64 bit ket addressing is not necessary. However, `-DQBCAPPOW=10` could be added to the above to support high-width stabilizer and Schmidt decomposition cases, with the appropriate build of the Boost headers for the toolchain.

## Compiling to C from WASM

Arbitrary Qrack executables will not all necessarily link, at this time. However, `wasm2c` can generate C code from Qrack executables and modules, and *some* will successfully link.

For example, generate a `.c` module and a `.h` header from your build/examples directory:

```sh
wasm2c teleport.wasm -o teleport.c
```

Compile and link it from `wabt`, (though the command below assumes that `wabt` is already in your path).

```sh
cc -o teleport -Iwabt/wasm2c/ teleport.c wabt/wasm2c/wasm-rt-impl.c -lm
```

We apologize for providing an example that will not quite work. However, with `emcc` or `cc`, this is the general idea. Once the generated `.c` and `.h` file are syntactically valid, via modification of your original Qrack C++ program, then static linkage must be specified with `-l`, assuming appropriate libraries for static linkage are actually available. (Emscripten is under active development, and we thank its maintainers.)

## Performing code coverage

```sh
    $ cd _build
    $ cmake -DENABLE_CODECOVERAGE=ON ..
    $ make -j 8 unittest
    $ ./unittest
    $ make coverage
    $ cd coverage_results
    $ python -m http.server
```

## Changing default OpenCL device
OpenCL device(s) can be specified by index in `Qrack::QInterface` subclass constructors. The global default device can also be overridden with the environment variable `QRACK_OCL_DEFAULT_DEVICE=n`, where `n` is the index of the OpenCL device you want to use, as reported by the OpenCL initialization header.

## Enable OpenCL device redistribution
Setting the environment variable `QRACK_ENABLE_QUNITMULTI_REDISTRIBUTE` to any value except a null string enables reactive load redistribution or balancing, in `QUnitMulti`. Otherwise, `QUnitMulti` only tries to balance load as opportunity arises when new separable `QEngineShard` instances are created.

## Tune OpenCL preferred concurrency
Preferred concurrency has a tunable offset with default value of `3`, with the environment variable setting `export QRACK_GPU_OFFSET_QB=[m]` for some (positive or negative) integer `m`. For each integer increment of `m`, the preferred concurrency is multiplied by 2. (Preferred concurrency is calculated as `pow2(ceil(log2(([GPU processing element count] * [preferred group size for the single qubit gate kernel, usually warp size])))) << QRACK_GPU_OFFSET_QB`.)

## QPager distributed simulation options
`QPager` attempts to smartly allocate low qubit widths for maximum performance. For heterogeneous GPU simulation, based on `clinfo`, you can set a ceiling on your maximum OpenCL accelerator state vector page allocation with the environment variable `QRACK_MAX_PAGE_QB=n`, where n is an integer >=0. The default `n` is max integer, meaning that maximum allocation segment of your GPU RAM is always a single hardware page.

To set a maximum on how many qubits can be allocated on a single `QPager` instance, use the environment variable `QRACK_MAX_PAGING_QB`, for example, `export QRACK_MAX_PAGING_QB=30` to cause `QPager` to throw an exception that can be caught if it is asked to allocate 31 or more qubits. 

To set the `QPager` device ID list, use the `QRACK_QPAGER_DEVICES` environment variable. This variable should contain an ordered list of Qrack OpenCL device IDs that should be automatically used in all `QPager` instances. Note that device IDs may be included multiple times in this list in order to achieve a simple form of load balancing. For example, since NVIDIA GPUs typically have 4 maximum allocation segments, a device list like `1,1,1,1,0,0,0,0` will allocate the first 4 maximum allocation segments on device `1` first, such that device `1` will be (roughly) 100% utilized before including any segments on device `0`. This list can also be written `4.1,4.0`, which means that 4 segments of `1` should be repeated before 4 segments of `0` are repeated. To repeat a pattern of multiple IDs, follow the multiplier with multiple `.` characters separating every ID in the pattern, like `4.1.0,4.2` for 4 repetitions of `1,0` followed by 4 repetitions of `2`. If device IDs are exhausted in the device list, `QPager` will automatically cycle the list as many times as it needs, to attempt higher segment count allocation.

There are two special device IDs that can be specified in these lists: `-1` is global Qrack default device. `-2` indicates that `QInterface`-local constructor-specified device ID should be used. (For example, a device list argument of just `-2` will indicate that distribution choices should defer to those of `QUnitMulti`, if in use.)

`QRACK_QPAGER_DEVICES_HOST_POINTER` corresponds to each device ID in `QRACK_QPAGER_DEVICES`, per sequential item in that other variable, (with the same syntax and list wrapping behavior). If the value of this is `0` for a page, that page attempts OpenCL _device_ RAM allocation; if the value is `1` for a page, that page attempts OpenCL _host_ RAM allocation. `0` value, device RAM, is suggested for GPUs; `1` value, host RAM, is suggested for CPUs and APUs (which use general host RAM, anyway). By default, all devices attempt on-device RAM allocation, if this environment variable is not specified.

## QUnitMulti device list
Specify a device list for `QUnitMulti` the same way you would for `QPager`, with environment variable `QRACK_QUNITMULTI_DEVICES`. Corresponding to each entry in `QRACK_QUNITMULTI_DEVICES`, use `QRACK_QUNITMULTI_DEVICES_MAX_QB` to (optionally) specify a per-entry ceiling on device usage. For smaller-width devices like CPUs, it might make sense to set the qubit ceiling to about the CPU `PSTRIDEPOW` plus logarithm base 2 of your hyperthread count.

## QTensorNetwork options
`QTensorNetwork` has a threshold up to which it is able to reuse more work in measurement and probability calculations, `QRACK_QTENSORNETWORK_THRESHOLD_QB`. Its default value is 30 qubits. Above (and not including) this threshold, `QTensorNetwork` will use techniques like restricting to "past light cones" for measurement and probablity calculation, in an attempt to reduce overall memory footprint at the cost of additional execution time.

## QBdt and QBdtHybrid options
`QBdtHybrid` sets a threshold for "hybridization" between "quantum binary decision diagrams" (see Acknowledgements at bottom of document) and state vector simulation, based on how efficiently the "diagram" or "tree" can be "compressed." The environment variable `QRACK_QBDT_HYBRID_THRESHOLD` (typically taking values between 0 and 1) sets a multiplicative fraction for maximally-compressed size of the tree, as fraction of node count vs. equivalent state vector amplitude count, before switching over to state vector simulation. Note that maximum `QBdt` node count is _twice_ the count of amplitudes in the equivalent state vector simulation, so set the variable to 2 or higher to completely suppress switching and recover `QBdt`-only simulation in all cases.

## Build and environment options for CPU engines
`QEngineCPU` and `QHybrid` batch work items in groups of 2^`PSTRIDEPOW` before dispatching them to single CPU threads, potentially greatly reducing waiting on mutexes without signficantly hurting utilization and scheduling. The default for this option can be controlled at build time, by passing `-DPSTRIDEPOW=n` to CMake, with "n" being an integer greater than or equal to 0. This can be overridden at run time by the enviroment variable `QRACK_PSTRIDEPOW=n`. If an environment variable is not defined for this option, the default from CMake build will be used. (The default is meant to work well across different typical consumer systems, but it might benefit from system-tailored tuning via the environment variable.)

`-DENABLE_QUNIT_CPU_PARALLEL=OFF` disables asynchronous dispatch of `QStabilizerHybrid` and low width `QEngineCPU`/`QHybrid` gates with `std::future`. This option is on by default. Typically, `QUnit` stays safely under maximum thread count limits, but situations arise where async CPU simulation causes `QUnit` to dispatch too many CPU threads for the operating system. This build option can also reduce overall thread usage when Qrack user code operates in a multi-threaded or multi-shell environment. (Linux thread count limits might be smaller than Windows.)

## Maximum allocation guard
Set the maximum allowed allocation (in MB) for the global OpenCL pool with `QRACK_MAX_ALLOC_MB`. Per OpenCL device, this sets each maximum allocation limit with the same syntax as `QRACK_QPAGER_DEVICES`. (Succesive entries in the list are MB limits numbered according to Qrack's device IDs print-out on launch.) By default, each device is capped at 3/4 of its available global memory, for stability in common use cases. This includes (VRAM) state vectors and auxiliary buffers larger than approximately `sizeof(bitCapIntOcl) * sizeof(bitCapIntOcl)`. This should also include out-of-place single duplication of any state vector. This does **not** include non-OpenCL general heap or stack allocation.

`QRACK_MAX_PAGING_QB` and `QRACK_MAX_CPU_QB` environment variables set a maximum on how many qubits can be allocated on a single `QPager` or `QEngineCPU` instance, respectively. This qubit limit is for maximum single `QPager` or `QEngineCPU` allocation, whereas `QUnit` and `QUnitMulti` might allocate _more_ qubits than this as separable subsystems, requiring that no individual separable subsystem exceeds the qubit limit environment variables. (`QEngineOCL` limits are automatically maximal according to a query Qrack makes of maximum allocation segment on a given OpenCL device.)

Note that this controls total direct Qrack OpenCL buffer allocation, not total Qrack library allocation. Total OpenCL buffer allocation is **not** fully indicative of total library allocation.

## Approximation options
`QUnit` can optionally round qubit subsystems proactively or on-demand to the nearest single or double qubit eigenstate with the `QRACK_QUNIT_SEPARABILITY_THRESHOLD=[0.0 - 1.0]` environment variable, with a value between `0.0` and `1.0`. When trying to find separable subsystems, Qrack will start by making 3-axis (independent or conditional) probability measurements. Based on the probability measurements, under the assumption that the state _is_ separable, an inverse state preparation to |0> procedure is fixed. If inverse state preparation would bring any single qubit Bloch sphere projection within parameter range of the edge of the Bloch sphere, (with unit length, `1.0`,) then the subsystem will be rounded to that state, normalized, and then "uncomputed" with the corresponding (forward) state preparation, effectively "hyperpolarizing" one and two qubit separable substates by replacing entanglement with local qubit Bloch sphere extent. (If 3-axis probability is _not_ within rounding range, nothing is done directly to the substate.)

Similarly functionality to above is available for `QBdt` with `QRACK_QBDT_SEPARABILITY_THRESHOLD=[0.0 - 0.5]`. In the case of this parameter, any branch with less than the parameter value for probability is rounded to 0, and its partner branch is renormalized to unit length. This same value is also used for branch equality comparison.

Environment variable `QRACK_NONCLIFFORD_ROUNDING_THRESHOLD` sets the non-Clifford phase gate magnitude, as a fraction of `T` gate phase angle, (from the closest Clifford state Bloch sphere orientation,) that will be rounded to 0 in terminal measurement and sampling operations. (For `0`/default value, all non-Clifford phase gates are exactly preserved.)

## Vectorization optimization

```sh
$ cmake -DENABLE_COMPLEX_X2=ON ..
```
Multiply complex numbers two at a time instead of one at a time. Requires AVX for double and SSE 1.0 (with optional SSE 3.0) for float. On by default, but can be turned off for double accuracy without the AVX requirement, or to completely remove vectorization with single float accuracy.

If `-DENABLE_COMPLEX_X2=ON`, then SSE 3.0 is used by default. Turn off the following option to limit to SSE 1.0 level:

```sh
$ cmake -DENABLE_SSE3=OFF ..
```
## Random number generation options (on-chip by default)

```sh
$ cmake -DENABLE_RDRAND=OFF ..
```
Turn off the option to attempt using on-chip hardware random number generation, which is on by default. If the option is on, Qrack might still compile to attempt using hardware random number generation, but fall back to software generation if the RDRAND opcode is not actually available. Some systems' compilers, such as that of the Raspberry Pi 3, do not recognize the compilation flag for enabling RDRAND, in which case this option needs to be turned off.

```sh
$ cmake [-DENABLE_RDRAND=OFF] -DENABLE_DEVRAND=ON ..
```
Instead of RDRAND, use Linux `/dev/urandom/` as the Qrack random number source. (The necessary system call will only be available on Linux systems.)

```sh
$ cmake -DSEED_DEVRAND=OFF ..
```
If pure software pseudo-random number generator is used, it will be seeded from `/dev/random` by default. `-DSEED_DEVRAND=OFF` will use the system clock for Mersenne twister seeding, instead of `/dev/random`.

## Pure 32 bit OpenCL kernels (including OpenCL on Raspberry Pi 3)

```sh
$ cmake -DENABLE_PURE32=ON ..
```
This option is needed for certain older or simpler hardware. This removes all use of 64 bit types from the OpenCL kernels, as well as completely removing the use of SIMD intrinsics. Note that this build option theoretically supports only up to 32 qubits, whereas `-DENABLE_PURE32=OFF` could support up to 64 qubits, (if the memory requirements were realistically attainable for either 32-bit or 64-bit hardware, or in limited cases available for `QUnit` Schmidt decomposition). `-DENABLE_PURE32=ON` is necessary to support the VC4CL OpenCL compiler for the VideoCore GPU of the Raspberry Pi 3. (Additionally, for that platform, the RDRAND instruction is not available, and you should `-DENABLE_RDRAND=OFF`. VC4CL for the VideoCore GPU is currently fully supported.)

## Reduced or increased coherent qubit addressing

```sh
$ cmake [-DUINTPOW=n] [-DQBCAPPOW=n] ..
```
Qrack uses an unsigned integer primitive for ubiquitous qubit masking operations, for "local" qubits (`QEngine`) and "global" qubits (`QUnit` and `QPager`). This limits the maximum qubit capacity of any coherent `QInterface` to the total number of bits in the global (or local) masking type. By default, a 64-bit unsigned integer is used, corresponding to a maximum of 64 qubits in any coherent `QInterface` (if attainable, such as in limited cases with `QUnit`). `-DUINTPOW=n` reduces the "local" masking type to 2^n bits, (ex.: for max OpenCL sub-unit or page qubit width,) which might also be important with accelerators that might not support 64-bit types. `-DQBCAPPOW=n` sets the maximum power of "global" qubits in "paged" or `QUnit` types as potentially larger than single "pages" or "sub-units," for "n" >= 5, with n=5 being 2^5=32 qubits. Large "n" is possible with the Boost big integer header. (Setting "n" the same for both build options can avoid casting between "subunit" and "global qubit" masking types, if larger "paging" or `QUnit` widths than `QEngine` types are not needed.)

## Variable floating point precision

```sh
$ cmake [-DFPPOW=n] ..
```
Like for unsigned integer masking types, this sets the floating point accuracy for state vectors to n^2. By default n=5, for 2^5=32 bit floating point precision. "half," "double," and "quad," availability depend on the system, but n=6 for "double" is commonly supported on modern hardware. n=4 for half is supported by GCC on ARM, header-only on x86_64, and by device pragma if available for OpenCL kernels. "quad" is supported on CPU only, if available.

## Precompiled OpenCL kernels

```sh
$ qrack_cl_compile [path]
```
Precompile the OpenCL programs for all available devices, and save them to the optional "path" parameter location. By default, programs will be saved to a folder in the "home" directory, such as `~/.qrack/` on most Linux systems. (The default path can also be specified as an environment variable, `QRACK_OCL_PATH`.) Also by default, Qrack will attempt to load precompiled binaries from the same path, but the library will fall back to JIT compilation if program binaries are not available or are corrupt. To turn off default loading of binaries, one can simply delete the programs from this folder.

The option to load and save precompiled binaries, and where to load them from, can be controlled with the initializing method of `Qrack::OCLEngine`:

```cpp
Qrack::OCLEngine::InitOCL(true, true, Qrack::OCLEngine::GetDefaultBinaryPath());
```
To use this method directly, _it needs to be called before any OpenCL simulators are created in the program,_ as initialization happens automatically upon creating any OpenCL simulator instance. Calling the `OCLEngine::InitOCL()` method directly also ensures that the singleton instance has been created, with the results of the initialization call. The initialization method prototype is as follows:

```cpp
/// Initialize the OCL environment, with the option to save the generated binaries. Binaries will be saved/loaded from the folder path "home".
static void InitOCL(bool buildFromSource = false, bool saveBinaries = false, std::string home = "*");
```
The `home` argument default indicates that the default home directory path should be used.

## VM6502Q

```sh
$ cmake -DENABLE_VM6502Q_DEBUG=ON ..
```
Qrack was originally written so that the disassembler of VM6502Q should show the classical expecation value of registers, following Ehrenfest's theorem. However, this incurs significant additional overhead for `QInterface::IndexedLDA()`, `QInterface::IndexedADC()`, and `QInterface::IndexedSBC()`. As such, this behavior in the VM6502Q disassembler is only supported when this CMake flag is specifically enabled. (It is off by default.) These three methods will return 0, if the flag is disabled.

## Turn on/off optional API components

```sh
$ cmake -DENABLE_BCD=OFF -DENABLE_REG_GATES=OFF -DENABLE_ROT_API=OFF -DENABLE_ALU=ON ..
```

Prior to the Qrack v7 API, a larger set of convenience methods were included in all builds, which increased the size of the library binary. By default, `ENABLE_REG_GATES`, `ENABLE_ROT_API` and  `ENABLE_BCD` all default to `OFF`, while `ENABLE_ALU` for arithmetic logic unit methods defaults to `ON`.

`ENABLE_REG_GATES` adds various looped-over-width gates to the API, like the lengthwise `CNOT(control, target, length)` method. This method is a convenience wrapper on a loop of `CNOT` operations for `length`, starting from `control` and `target`, to `control + length - 1` and `target + length - 1`. These methods were seen as opportunities for optimization, at a much earlier point, but they have fallen out of internal use, and basically none of them are optimized as special cases, anymore. Disabling `ENABLE_REG_GATES` does **not** remove lengthwise `X(target, length)` and `H(target, length)` methods, as these specific convenience methods are still commonly used in the protected API, for negating or superposing across register width.

`ENABLE_ROT_API` adds many less common **rotation** methods to the API, like dyadic fraction rotations. These never found common use in the protected API, while they add significant size to compiled binaries.

"BCD" arithmetic ("binary coded decimal") is necessary to support emulation based on the MOS-6502. However, this is an outmoded form of binary arithmetic for most or all conceivable purposes for which one would want a quantum computer. (It stores integers as base 10 digits, in binary.) Off by default, turning this option on will slightly increase binary size by including BCD ALU operations from the API, but this is necessary to support the VM6502Q chip-like emulator project.

## Copyright, License, and Acknowledgements

Copyright (c) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.

Daniel Strano would like to specifically note that Benn Bollay is almost entirely responsible for the initial implementation of `QUnit` and tooling, including unit tests, in addition to large amounts of work on the documentation and many other various contributions in intensive reviews. Special thanks go to Aryan Blaauw for his extensive systematic benchmark program, leading to much debugging and design feedback, while he spreads general good will about our community discussion space. Also, thank you to Marek Karcz for supplying an awesome base classical 6502 emulator for proof-of-concept. For unit tests and benchmarks, Qrack uses Catch v2.13.7 under the Boost Software License, Version 1.0. The `QStabilizer` partial simulator "engine" is adapted from CHP by Scott Aaronson, for non-commercial use. `QBdt` is Qrack's "hand-rolled" take on "quantum binary decision diagrams" ("QBDD," or "quantum binary decision trees") inspired largely by a talk Dan attended from Jülich Supercomputing Center at IEEE Quantum Week, in 2021, and later followed up with reading into work of authors including Robert Wille. Half precision floating point headers are provided by [http://half.sourceforge.net/](http://half.sourceforge.net/), with our thanks. GitHub user [paniash](https://github.com/paniash) has kindly contributed `README.md` styling and standardization. Some commits might be written with the assistance of OpenAI's ChatGPT, though the commit messages should note all such specific cases, and 0 commits used direct ChatGPT assistance or any direct AI assistance for authorship before April 15, 2023. Thank you to all our PR contributors, tracked in GitHub, and thank you to the OSS community in general for supporting code, including [Adam Kelly](https://github.com/libtangle/qcgpu) and the [qulacs team](https://github.com/qulacs/qulacs), for Qiskit and Cirq interfaces. (Additionally, the font for the Qrack logo is "Electrickle," distributed as "Freeware" from [https://www.fontspace.com/fontastic/electrickle](https://www.fontspace.com/fontastic/electrickle).)

We thank the Unitary Fund for its generous support, in a project to help standardize benchmarks across quantum computer simulator software!  Thank you to any and all contributors!

Licensed under the GNU Lesser General Public License V3.

See [LICENSE.md](https://github.com/vm6502q/qrack/blob/main/LICENSE.md) in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.
