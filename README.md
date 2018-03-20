## Qrack

Copyright (c) Daniel Strano 2017, (with many thanks to Benn Bollay for tool chain development in particular, and also Marek Karcz for supplying an awesome base classical 6502 emulator for proof-of-concept). All rights reserved. (See "par_for.hpp" for additional information.)
Licensed under the GNU General Public License V3.
See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html for details.

This is a multithreaded framework for developing classically emulated virtual universal quantum processors. (See the doxygen entry for "CoherentUnit" for an outline of the algorithms by which Qrack is implemented.)

The intent of "Qrack" is to provide a framework for developing classically emulated universal quantum virtual machines. In addition to quantum gates, Qrack provides optimized versions of multi-bit, register-wise, opcode-like "instructions." A chip-like quantum CPU (QCPU) is instantiated as a "Qrack::CoherentUnit," assuming all the memory in the quantum memory in the QCPU is quantum mechanically "coherent" for quantum computation.

A CoherentUnit can be thought of as like simply a one-dimensional array of qubits. Bits can manipulated on by a single bit gate at a time, or gates and higher level quantum instructions can be acted over arbitrary contiguous sets of bits. A qubit start index and a length is specified for parallel operation of gates over bits or for higher level instructions, like arithmetic on abitrary width registers. Some methods are designed for (bitwise and register-like) interface between quantum and classical bits. See the Doxygen for the purpose of gate-like and register-like functions.

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
make all
```

Instantiate a Qrack::CoherentUnit, specifying the desired number of qubits. (Optionally, also specify the initial bit permutation state in the constructor.) Coherent units can be "cohered" and "decohered" with each other, to simulate coherence and loss of coherence of separable subsystems between distinct quantum bit registers. Both single quantum gate commands and register-like multi-bit commands are available.

For more information, compile the doxygen.config in the root folder, and then check the "doc" folder.

The included EXAMPLE.CPP is headed by a unit tests. Then, the following example is run:

This is a simple example of quantum mechanics simulation in quantum computational logic. It is essentially a unidirectional binary quantum random walk algorithm, from a positive starting point, heading toward zero.

We assume a fixed length time step. During each time step, we step through an equal superposition of either standing still or taking one fixed length step from our current position toward our fixed destination.

This is equivalent to a physical body having a 50% chance of emitting a fixed unit of energy per a fixed unit of time, in a pure quantum state. Hence, it might be considered a simple quantum mechanics simulation.

## Installing OpenCL on VMWare

1.  Download the [AMD APP SDK](https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)
1.  Install it.
1.  Add symlinks for `/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/libOpenCL.so.1` to `/usr/lib`
1.  Add symlinks for `/opt/AMDAPPSDK-3.0/lib/x86_64/sdk/libamdocl64.so` to `/usr/lib`
1.  Make sure `clinfo` reports back that there is a valid backend to use (anything other than an error should be fine).
1.  Adjust the `makefile` to have the appropriate search paths

## Note about unitarity and arithmetic

The project's author notes that the ADD and SUB variants in the current version of the project break unitarity and are not on rigorous footing. They are included in the project as the basis for work on correct implementations. INC and DEC, however, are unitary and function much like SWAP operations. ADD and SUB operations are provisional and will be corrected.

Similarly, AND/OR/XOR are only provided for convenience and generally entail a measurement of the output bit. For register-based virtual quantum processors, we suspect it will be a common requirement that an output register be measured, cleared, and loaded with the output logical comparison operations, but this will generally require measurement and therefore introduce a random phase factor. CCNOT and X gates (composed for convenience as "AntiCCNOT" gates) could instead be operated on a target bit with a known input state to achieve a similar result in a unitary fashion, but this is left to the particular virtual machine implementation.

Similarly, the "Decohere" and "Dispose" methods should only be used on qubits that are guaranteed to be separable.

Qrack is an experimental work in progress, and the author aims for both utility and correctness, but the project cannot be guaranteed to be fit for any purpose, express or implied. (See LICENSE.md for details.)
