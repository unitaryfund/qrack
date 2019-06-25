# Qrack

[![Qrack Build Status](https://api.travis-ci.org/vm6502q/qrack.svg?branch=master)](https://travis-ci.org/vm6502q/qrack/builds) [![Mentioned in Awesome awesome-quantum-computing](https://awesome.re/mentioned-badge.svg)](https://github.com/desireevl/awesome-quantum-computing)

This is a multithreaded framework for developing classically emulated virtual universal quantum processors. It has CPU, GPU, and multi-processor engine types.

The intent of "Qrack" is to provide a framework for developing practical, computationally efficient, classically emulated universal quantum virtual machines. In addition to quantum gates, Qrack provides optimized versions of multi-bit, register-wise, opcode-like "instructions." A chip-like quantum CPU (QCPU) is instantiated as a "Qrack::QUnit." "Qrack::QEngineCPU" and "Qrack::QEngineOCL" represent fully entangled cases and underlie "Qrack::QUnit."

A QUnit or QEngine can be thought of as like simply a one-dimensional array of qubits. Bits can manipulated on by a single bit gate at a time, or gates and higher level quantum instructions can be acted over arbitrary contiguous sets of bits. A qubit start index and a length is specified for parallel operation of gates over bits or for higher level instructions, like arithmetic on abitrary width registers. Some methods are designed for (bitwise and register-like) interface between quantum and classical bits. See the Doxygen for the purpose of gate-like and register-like functions.

Qrack has already been integrated with a MOS 6502 emulator, (see https://github.com/vm6502q/vm6502q,) which demonstrates Qrack's primary purpose. (The base 6502 emulator to which Qrack was added for that project is by Marek Karcz, many thanks to Marek! See https://github.com/makarcz/vm6502.)

Virtual machines created with Qrack can be abstracted from the code that runs on them, and need not necessarily be packaged with each other. Qrack can be used to create a virtual machine that translates opcodes into instructions for a virtual quantum processor. Then, software that runs on the virtual machine could be abstracted from the machine itself as any higher-level software could. All that is necessary to run the quantum binary is any virtual machine with the same instruction set.

Direct measurement of qubit probability and phase are implemented for debugging, but also for potential speed-up, by allowing the classical emulation to leverage nonphysical exceptions to quantum logic, like by cloning a quantum state. In practice, though, opcodes might not rely on this (nonphysical) functionality at all. (The MOS 6502 emulator referenced above does not.)

Qrack compiles like a library. To include in your project:

1. In your source code:
```
#include "qregister.hpp"
```

2. On the command line, in the project directory

```
$ mkdir _build && cd _build && cmake .. && make all install
```

Instantiate a Qrack::QUnit, specifying the desired number of qubits. (Optionally, also specify the initial bit permutation state in the constructor.) QUnits can be "cohered" and "decohered" with each other, to simulate coherence and loss of coherence of separable subsystems between distinct quantum bit registers. Both single quantum gate commands and register-like multi-bit commands are available.

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

Most platforms offer a standardized way of installing OpenCL, however for VMWare it's a little more peculiar.

1.  Download the [AMD APP SDK](https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)
1.  Install it.
1.  Add symlinks for `/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/libOpenCL.so.1` to `/usr/lib`
1.  Add symlinks for `/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/libamdocl64.so` to `/usr/lib`
1.  Make sure `clinfo` reports back that there is a valid backend to use (anything other than an error should be fine).
1.  Install OpenGL headers: `$ sudo apt install mesa-common-dev`
1.  Adjust the `makefile` to have the appropriate search paths

## Installing OpenCL on Mac

While the OpenCL framework is available by default on most modern Macs, the C++ header “cl.hpp” is usually not. One option for building for OpenCL is to download this header file and include it in include/OpenCL (as “cl.hpp”). The OpenCL C++ header can be found at the Khronos OpenCL registry:

https://www.khronos.org/registry/OpenCL/

## Building and Installing Qrack on Windows

Qrack supports building on Windows, but some special configuration is required. Windows 10 usually comes with default OpenCL libraries for Intel (or AMD) CPUs and their graphics coprocessors, but NVIDIA graphics card support might require the CUDA Toolkit. The CUDA Toolkit also provides an OpenCL development environment, which is generally necessary to build Qrack.

Qrack requires the `xxd` command to convert its OpenCL kernel code into hexadecimal format for building. `xxd` is not natively available on Windows systems, but Windows executables for it are provided by sources including the [Vim editor Windows port](https://www.vim.org/download.php).

CMake on Windows will set up a 32-bit Visual Studio project by default, (if using Visual Studio). Putting together all of the above considerations, after installing the CUDA Toolkit and Vim, a typical CMake command for Windows might look like this:

```
    $ mkdir _build
    $ cd _build
    $ cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DXXD_BIN="C:/Program Files (x86)/Vim/vim81/xxd.exe" ..
```

After CMake, the project must be built in Visual Studio.

## Performing code coverage

```
    $ cd _build
    $ cmake -DENABLE_CODECOVERAGE=ON ..
    $ make -j 8 unittest
    $ ./unittest
    $ make coverage
    $ cd coverage_results
    $ python -m SimpleHTTPServer
```

## Vectorization optimization

```
$ cmake -DENABLE_COMPLEX_X2=ON ..
```
Multiply complex numbers two at a time instead of one at a time. Requires AVX for double and SSE 1.0 for float. On by default, but can be turned off for double accuracy without the AVX requirement, or to completely remove vectorization with single float accuracy.

## Reduce accuracy from double to float

```
$ cmake -DENABLE_COMPLEX8=ON ..
```
Reduce to float accuracy for complex numbers. Requires half as much RAM (1 additional qubit). Compatible with SSE 1.0 and single precision accelerator devices.

## On-Chip Hardware Random Number Generation 

```
$ cmake -DENABLE_RDRAND=OFF ..
```
Turn off the option to attempt using on-chip hardware random number generation, which is on by default. If the option is on, Qrack might still compile to attempt using hardware random number generation, but fall back to software generation if the RDRAND opcode is not actually available. Some systems' compilers, such as that of the Raspberry Pi 3, do not recognize the compilation flag for enabling RDRAND, in which case this option needs to be turned off.

## Pure 32 bit OpenCL kernels (including Raspberry Pi 3)

```
$ cmake -DENABLE_PURE32=ON ..
```
This option is needed for certain older or simpler hardware. This removes all use of 64 bit types from the OpenCL kernels, as well as completely removing the use of SIMD intrinsics. Note that this build option theoretically supports only up to 32 qubits, whereas `-DENABLE_PURE32=OFF` could support up to 64 qubits, (if the memory requirements were realistically attainable for either 32-bit or 64-bit hardware). `-DENABLE_PURE32=ON` should be necessary but sufficient to support the VC4CL OpenCL compiler for the VideoCore GPU of the Raspberry Pi 3.

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

## Pure 32 bit OpenCL kernels (including Raspberry Pi 3)

```
$ cmake -DENABLE_VM6502Q_DEBUG=ON ..
```
Qrack was originally written so that the disassembler of VM6502Q should show the classical expecation value of registers, following Ehrenfest's theorem. However, this incurs significant additional overhead for `QInterface::IndexedLDA()`, `QInterface::IndexedADC()`, and `QInterface::IndexedSBC()`. As such, this behavior in the VM6502Q disassembler is only supported when this CMake flag is specifically enabled. (It is off by default.) These three methods will return 0, if the flag is disabled.

## Copyright and License

Copyright (c) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.

Daniel Strano would like to specifically note that Benn Bollay is almost entirely responsible for the implementation of QUnit and tooling, including unit tests, in addition to large amounts of work on the documentation and many other various contributions in intensive reviews. Also, thank you to Marek Karcz for supplying an awesome base classical 6502 emulator for proof-of-concept. (Additionally, the font for the Qrack logo is "Electrickle," distributed as "Freeware" from [https://www.fontspace.com/fontastic/electrickle](https://www.fontspace.com/fontastic/electrickle).) Thank you to any and all contributors!

Licensed under the GNU Lesser General Public License V3.

See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.
