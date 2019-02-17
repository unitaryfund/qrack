# Changelog

**Thank you to [tbabej and nlewycky](https://github.com/vm6502q/qrack/graphs/contributors) for contributing to the documentation!**

**Thank you to [twobombs](https://github.com/twobombs) for testing the OSX Mojave build!**

## vm6502q.v3.2: Windows 10 and Mac support, and ApproxCompare() fix

This build debugs support on Mac Mojave, adds support for building with Visual Studio on Windows 10, and fixes a longstanding bug in the ApproxCompare() method, which had been leading to unit test failures. The changes in code itself are minor, but since they add wide platform support, and stable general platform support was the aim of v3.1, we're tagging this version before any new functionality is added.

## vm6502q.v3.1: General hardware support and stack-readiness

This release improves stability across all tested hardware platforms and debugs support for optional integration with the vm6502q fork of ProjectQ.

**What's new in Qrack v3.1**

- Supports building on the Raspberry Pi 3 with VC4CL
- Optional integration with vm6502q/ProjectQ fork has been completed
- "Uniformly controlled" gates added to the API
- Bug fix in TimeEvolve() method
- General stability improvements across many different platforms

**Known Issues**

ApproxCompare() has a sensitive threshold for state comparison, which causes unit tests based on it to fail sporadically. TryDecompose() and TrySeparate() also use ApproxCompare(). A low threshold means more conservative behavior. TryDecompose() and TrySeparate() are less likely to give false positives for true separability, but they are more likely to give false negatives.


## vm6502q.v3.0: API and performance improvements, gate fusion, and ProjectQ compatibility

This release changes the API to allow arbitrary numbers of control bits, adds a gate fusion layer called Qrack::QFusion, and makes major improvements to speed and memory usage.

OpenCL single gates now dispatch entirely asynchronously, returning to the main thread after adding gates to the OpenCL queue, so that work can continue on the CPU without blocking.

If possible, by default, Qrack will attempt to allocate its state vectors entirely on OpenCL device memory, which can recover up to about 1 or 2 GB of general heap for a small enough state vector, for many modern accelerators.

General performance improvements, including fewer redundant normalizations, have resulted in a large factor of improvement in overall speed, without sacrificing accuracy.

Bugs were fixed in QUnit, for edge cases for logic pertaining to explicit separability of qubits.

New methods have been added to help support quantum mechanics simulation, including the "TimeEvolve" method and a matrix exponentiation gate. The API has been generally updated to help support integration as a back end for ProjectQ. (https://github.com/ProjectQ-Framework/ProjectQ)

## vm6502q.v2.0: OpenCL Performance Improvements

**What’s New in Qrack v2.0**

- Greatly improved all-around OpenCL performance (See https://qrack.readthedocs.io/en/two_point_zero/performance.html)
- "Full OpenCL coverage” - QEngineOCL no longer inherits from QEngineCPU at all, and QEngineOCL state vector manipulations are virtually entirely done with OpenCL kernels and the OpenCL API.
- Operator exponentiation methods have been added to the public API, (“Exp,” “ExpX,” “ExpY,” etc..) as well as a single bit gate method with a 2x2 complex matrix specified arbitrarily by the user
- Experimental multi-processor engine, QEngineOCLMulti
- Better explicit qubit separation in QUnit (less RAM, often greater speed, depending on use case)
- Tested and debugged for single and multi-processor compatibility with the Intel HD. Issues diagnosed and fixed for the HD include OpenCL compilation for single-accuracy-float-only devices, as well as logical compatibility of kernel calls with an arbitrary number of processing elements on a device, as opposed to an exact power of 2 processing elements.
- General minor bug fixes, (including small memory leaks, bad OpenCL group sizes, and others)

## vm6502q.v1.7: Public Release

This release of the Qrack Quantum Simulator and the associated VM6502Q toolchain is suitable for public use and large scale deployment in production.
