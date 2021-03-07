# Qrack

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3369483.svg)](https://doi.org/10.5281/zenodo.3369483) [![Qrack Build Status](https://api.travis-ci.org/vm6502q/qrack.svg?branch=main)](https://travis-ci.org/vm6502q/qrack/builds) [![Mentioned in Awesome awesome-quantum-computing](https://awesome.re/mentioned-badge.svg)](https://github.com/desireevl/awesome-quantum-computing)

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

The open source vm6502q/qrack library and its associated plugins and projects under the vm6502q organization header comprise a framework for full-stack quantum computing development, via high performance and fundamentally optimized simulation. The intent of "Qrack" is to provide maximum performance for the simulation of an ideal, virtually error-free quantum computer, across the broadest possible set of hardware and operating systems.

Using the C++11 standard, at base, Qrack has an external-dependency-free CPU simulator "engine," as well as a GPU simulator engine that depends only on OpenCL. The "QUnit" layer provides novel, fundamental optimizations in the simulation algorithm, based on "[Schmidt decomposition](https://arxiv.org/abs/1710.05867)," transformation of basis, 2 qubit controlled gate buffer caching, the physical nonobservability of arbitrary global phase factors on a state vector, and many other "synergistic" and incidental points of optimization between these approaches and in addition to them. "QUnit" can be placed "on top" of either CPU, GPU, or hybrid engine types, and an additional "QPager" layer can sit between these, or in place of QUnit. Optimizations and hardware support are highly configurable, particularly at build time.

A QUnit or QEngine can be thought of as like simply a one-dimensional array of qubits, within which any qubit has the capacity to directly and fully entangle with any and all others. Bits can be manipulated on by a single bit gate at a time, or gates and higher level quantum instructions can be acted over arbitrary contiguous sets of bits. A qubit start index and a length is specified for parallel operation of gates over bits or for higher level instructions, like arithmetic on abitrary width registers. Some methods are designed for (bitwise and register-like) interface between quantum and classical bits. See the Doxygen for the purpose of gate-like and register-like functions.

Qrack has already been integrated with a [MOS 6502 emulator](https://github.com/vm6502q/vm6502q), which demonstrates Qrack's original purpose, for use in developing chip-like quantum computer emulators. (The base 6502 emulator to which Qrack was added for that project is by Marek Karcz, many thanks to Marek! See https://github.com/makarcz/vm6502.)

A number of useful "pseudo-quantum" operations, which could not be carried out by true hardware quantum computers easily or at all, are included in the API for purposes like debugging, but also for potential speed-up, by allowing the classical emulation to leverage nonphysical exceptions to quantum logic, like by cloning a quantum state. In practice, though, quantum circuit programs might not rely on this (nonphysical) functionality at all. (The MOS 6502 emulator referenced above does not. It happens, many of these methods will not be exotic at all, to those familiar with other major quantum computing libraries. For example, notably, simply returning a full vector of probability amplitudes actually qualifies as "pseudo-quantum," in this sense.)

Qrack compiles like a library. To include in your project:

1. In your source code:
```
#include "qrack/qfactory.hpp"
```

2. On the command line, in the project directory

```
$ mkdir _build && cd _build && cmake .. && make all install
```

Instantiate a Qrack::QUnit, specifying the desired number of qubits. (Optionally, also specify the initial bit permutation state in the constructor.) QUnits can be (Schmidt) "composed" and "decomposed" with and from each other, to join and separate the representations of qubit "registers" that are not entangled at the point (de)composition. Both single quantum gate commands and register-like multi-bit commands are available.

For distributed simulation, the Qrack::QPager layer will segment a single register into a power-of-two count of equal length pages, running on an arbitrary number of OpenCL accelerators. The QPager layer also scales to arbitrarily small as well as large qubit counts, such that it can be appropriate for use on a single accelerator for small width simulations. The QPager layer is also compatible with Clifford set preamble circuits simulated with QStabilizerHybrid, as a layer over QPager, and QHybrid for CPU/GPU switching can be used as the "engine" layer under it. For Qrack in a cluster environment, we support the SnuCL and VirtualCL OpenCL virtualization layers, with OpenCL v1.1 compliant host code without required "host pointers."

For more information, compile the doxygen.config in the root folder, and then check the "doc" folder.

## Documentation

Live version of the documentation, including API reference, can be obtained at: https://qrack.readthedocs.io/en/latest/

## Community

Qrack has a community home at the Advanced Computing Topics server on Discord, at: https://discordapp.com/invite/Gj3CHDy

For help getting started with contributing, see our [CONTRIBUTING.md](https://github.com/vm6502q/qrack/blob/doc_resources/CONTRIBUTING.md).

## test/tests.cpp

The included `test/tests.cpp` contains unit tests and usage examples. The unittests themselves can be executed:

```
    $ _build/unittest
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

While the OpenCL framework is available by default on most modern Macs, the C++ header “cl.hpp” is usually not. One option for building for OpenCL on Mac is to download this header file and include it in the Qrack project folder under include/OpenCL (as “cl.hpp”). The OpenCL C++ header can be found at the Khronos OpenCL registry:

https://www.khronos.org/registry/OpenCL/

## Building and Installing Qrack on Windows

Qrack supports building on Windows, but some special configuration is required. Windows 10 usually comes with default OpenCL libraries for Intel (or AMD) CPUs and their graphics coprocessors, but NVIDIA graphics card support might require the CUDA Toolkit. The CUDA Toolkit also provides an OpenCL development environment, which is generally necessary to build Qrack.

Qrack requires the `xxd` command to convert its OpenCL kernel code into hexadecimal format for building. `xxd` is not natively available on Windows systems, but Windows executables for it are provided by sources including the [Vim editor Windows port](https://www.vim.org/download.php).

CMake on Windows will set up a 32-bit Visual Studio project by default, (if using Visual Studio,) whereas 64-bit will probably be typically desired. Putting together all of the above considerations, after installing the CUDA Toolkit and Vim, a typical CMake command for Windows might look like this:

```
    $ mkdir _build
    $ cd _build
    $ cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DFPPOW=6 -DXXD_BIN="C:/Program Files (x86)/Vim/vim82/xxd.exe" ..
```

After CMake, the project must be built in Visual Studio. (`-DFPPOW=6` disables single `float` accuracy in favor of `double`, which should usually be used for building the Q# runtime with `QrackSimulator`.)

## Performing code coverage

```
    $ cd _build
    $ cmake -DENABLE_CODECOVERAGE=ON ..
    $ make -j 8 unittest
    $ ./unittest
    $ make coverage
    $ cd coverage_results
    $ python -m http.server
```

## QPager distributed simulation options
QPager attempts to smartly allocate low qubit widths for maximum performance. For wider qubit simulations, based on `clinfo`, you can segment your maximum OpenCL accelerator state vector page allocation into global qubits with the environment variable `QRACK_SEGMENT_GLOBAL_QB=n`, where n is an integer >=0. The default n is 0, meaning that maximum allocation segment of your GPU RAM is a single page. (For 1 global qubit, one segment would have 2 pages, akin to 2 single amplitudes, therefore one "global qubit," or 4 pages for n=2, because 2^2=4, etc., by exponent.)

`QRACK_DEVICE_GLOBAL_QB=n`, alternatively, lets the user also choose the performance "hint" for preferred global qubits per device. By default, n=2, for 2 global qubits or equivalently 4 pages per device. Despite the "hint," `QPager` will allocate fewer pages per OpenCL device for small-enough widths, to keep processing elements better occupied. Also, `QPager` will allocate more qubits than the hint, per device, if the maximum allocation segment is exceeded as specified by `QRACK_SEGMENT_GLOBAL_QB`.

## Vectorization optimization

```
$ cmake -DENABLE_COMPLEX_X2=ON ..
```
Multiply complex numbers two at a time instead of one at a time. Requires AVX for double and SSE 1.0 for float. On by default, but can be turned off for double accuracy without the AVX requirement, or to completely remove vectorization with single float accuracy.

## Increase accuracy from float to double

```
$ cmake -DENABLE_COMPLEX8=OFF ..
```
By default, Qrack builds for float accuracy. Turning the above option off increases to double accuracy for complex numbers. Requires twice as much RAM (basically reducing maximum by 1 available qubit, for QEngine types). Compatible with SSE 1.0 and single precision accelerator devices.

## On-Chip Hardware Random Number Generation 

```
$ cmake -DENABLE_RDRAND=OFF ..
```
Turn off the option to attempt using on-chip hardware random number generation, which is on by default. If the option is on, Qrack might still compile to attempt using hardware random number generation, but fall back to software generation if the RDRAND opcode is not actually available. Some systems' compilers, such as that of the Raspberry Pi 3, do not recognize the compilation flag for enabling RDRAND, in which case this option needs to be turned off.

## Pure 32 bit OpenCL kernels (including OpenCL on Raspberry Pi 3)

```
$ cmake -DENABLE_PURE32=ON ..
```
This option is needed for certain older or simpler hardware. This removes all use of 64 bit types from the OpenCL kernels, as well as completely removing the use of SIMD intrinsics. Note that this build option theoretically supports only up to 32 qubits, whereas `-DENABLE_PURE32=OFF` could support up to 64 qubits, (if the memory requirements were realistically attainable for either 32-bit or 64-bit hardware, or in limited cases available for QUnit Schmidt decomposition). `-DENABLE_PURE32=ON` is necessary to support the VC4CL OpenCL compiler for the VideoCore GPU of the Raspberry Pi 3. (Additionally, for that platform, the RDRAND instruction is not available, and you should `-DENABLE_RDRAND=OFF`. VC4CL for the VideoCore GPU is currently fully supported.)

## Reduced or increased coherent qubit addressing

```
$ cmake [-DUINTPOW=n] [-DQBCAPPOW=n] ..
```
Qrack uses an unsigned integer primitive for ubiquitous qubit masking operations, for "local" qubits (`QEngine`) and "global" qubits (`QUnit` and `QPager`). This limits the maximum qubit capacity of any coherent QInterface to the total number of bits in the global (or local) masking type. By default, a 64-bit unsigned integer is used, corresponding to a maximum of 64 qubits in any coherent `QInterface` (if attainable, such as in limited cases with `QUnit`). `-DUINTPOW=n` reduces the "local" masking type to 2^n bits, (ex.: for max OpenCL sub-unit or page qubit width,) which might also be important with accelerators that might not support 64-bit types. `-DQBCAPPOW=n` sets the maximum power of "global" qubits in "paged" or `QUnit` types as potentially larger than single "pages" or "sub-units," for "n" >= 5, with n=5 being 2^5=32 qubits. Large "n" is possible with the Boost big integer header. (Setting "n" the same for both build options can avoid casting between "subunit" and "global qubit" masking types, if larger "paging" or `QUnit` widths than `QEngine` types are not needed.)

## Variable floating point precision

```
$ cmake [-FPPOW=n] ..
```
Like for unsigned integer masking types, this sets the floating point accuracy for state vectors to n^2. By default n=5, for 5^2=32 bit floating point precision. "half" and "double" availability depend on the system, but n=6 for "double" is commonly supported on modern hardware, and n=4 is supported on ARM, which has a native "half" type.

## Precompiled OpenCL kernels

```
$ qrack_cl_compile [path]
```
Precompile the OpenCL programs for all available devices, and save them to the optional "path" parameter location. By default, programs will be saved to a folder in the "home" directory, such as `~/.qrack/` on most Linux systems. (The default path can also be specified as an environment variable, `QRACK_OCL_PATH`.) Also by default, Qrack will attempt to load precompiled binaries from the same path, but the library will fall back to JIT compilation if program binaries are not available or are corrupt. To turn off default loading of binaries, one can simply delete the programs from this folder.

The option to load and save precompiled binaries, and where to load them from, can be controlled with the initializing method of `Qrack::OCLEngine`:
```
Qrack::OCLEngine::InitOCL(true, true, Qrack::OCLEngine::GetDefaultBinaryPath());
```
Calling the `OCLEngine::InitOCL()` method directly also ensures that the singleton instance has been created, with the results of the initialization call. The initialization method prototype is as follows:
```
/// Initialize the OCL environment, with the option to save the generated binaries. Binaries will be saved/loaded from the folder path "home".
static void InitOCL(bool buildFromSource = false, bool saveBinaries = false, std::string home = "*");
```
The `home` argument default indicates that the default home directory path should be used. 

## VM6502Q

```
$ cmake -DENABLE_VM6502Q_DEBUG=ON ..
```
Qrack was originally written so that the disassembler of VM6502Q should show the classical expecation value of registers, following Ehrenfest's theorem. However, this incurs significant additional overhead for `QInterface::IndexedLDA()`, `QInterface::IndexedADC()`, and `QInterface::IndexedSBC()`. As such, this behavior in the VM6502Q disassembler is only supported when this CMake flag is specifically enabled. (It is off by default.) These three methods will return 0, if the flag is disabled.

## Copyright, License, and Acknowledgements

Copyright (c) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.

Daniel Strano would like to specifically note that Benn Bollay is almost entirely responsible for the initial implementation of QUnit and tooling, including unit tests, in addition to large amounts of work on the documentation and many other various contributions in intensive reviews. Also, thank you to Marek Karcz for supplying an awesome base classical 6502 emulator for proof-of-concept. For unit tests and benchmarks, Qrack uses Catch v2.13.2 under the Boost Software License, Version 1.0. The `QStabilizer` partial simulator "engine" is adapted from CHP by Scott Aaronson, for non-commercial use. (Additionally, the font for the Qrack logo is "Electrickle," distributed as "Freeware" from [https://www.fontspace.com/fontastic/electrickle](https://www.fontspace.com/fontastic/electrickle).)

We thank the Unitary Fund for its generous support, in a project to help standardize benchmarks across quantum computer simulator software!  Thank you to any and all contributors!

Licensed under the GNU Lesser General Public License V3.

See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.
