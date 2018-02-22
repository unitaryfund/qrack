Copyright (c) Daniel Strano 2017. All rights reserved. (See "par_for.hpp" for additional information.)
Licensed under the GNU General Public License V3.
See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html for details.

This is a multithreaded framework for developing virtual universal quantum processors.

The intent of "Qrack" is to provide a framework for developing classically emulated universal quantum virtual machines. In addition to quantum gates, Qrack provides optimized versions of multi-bit, register-wise, opcode-like "instructions." A chip-like quantum CPU (QCPU) is instantiated as a "Qrack::CoherentUnit," assuming all the memory in the quantum memory in the QCPU is quantum mechanically "coherent" for quantum computation.

A CoherentUnit can be thought of as like simply a one-dimensional array of qubits. Bits can manipulated on by a single bit gate at a time, or gates and higher level quantum instructions can be acted over arbitrary contiguous sets of bits. A qubit start index and a length is specified for parallel operation of gates over bits or for higher level instructions, like arithmetic on abitrary width registers. Some methods are designed for (bitwise and register-like) interface between quantum and classical bits. See the Doxygen for the purpose of gate-like and register-like functions.

Qrack has already been integrated with a MOS 6502 emulator, (see https://github.com/WrathfulSpatula/vm6502 ) which demonstrates Qrack's primary purpose. (The base 6502 emulator to which Qrack was added for that project is by Marek Karcz, many thanks to Marek! See https://github.com/makarcz/vm6502 .)

Virtual machines created with Qrack can be abstracted from the code that runs on them, and need not necessarily be packaged with each other. Qrack can be used to create a virtual machine that translates opcodes into instructions for a virtual quantum processor. Then, software that runs on the virtual machine could be abstracted from the machine itself as any higher-level software could. All that is necessary to run the quantum binary is any virtual machine with the same instruction set.

Direct measurement of qubit probability and phase are implemented for debugging, but also for potential speed-up, by allowing the classical emulation to leverage nonphysical exceptions to quantum logic, like by cloning a quantum state. In practice, though, opcode might not rely on this (nonphysical) functionality at all. (The MOS 6502 emulator referenced above does not.)

Qrack compiles like a library. To include in your project:

1)In your source code:
#include "qrack.hpp"
2)On the command line, in the project directory
make all

Make will link against math, pthreads, and OpenCL. Modify the makefile as necessary for inclusion in your project. Set the makefile QRACKVER variable to "qrack_ocl.cpp" for the OpenCL version, "qrack.cpp" for CPU parallelism only (without OpenCL), or "qrack_serial.cpp" for a totally non-parallel version. If you aren't using the OpenCL version, you should remove OpenCL linkage from the makefile. 

Instantiate a Qrack::CoherentUnit, specifying the desired number of qubits. (Optionally, also specify the initial bit state in the constructor.) Coherent units can be "cohered" and "decohered" with each other, to simulate coherence and loss of coherence between distinct quantum bit registers. Both single quantum gate commands and register-like multi-bit commands are available.

If you are using the OpenCL version of the headers, the OpenCL platform and device can be selected in the first call to the OpenCL context instance. (See "example.cpp" for details. The OpenCL context is managed by a singleton, shared between an arbitrary number of CoherentUnit objects.)

For more information, compile the doxygen.config in the root folder, and then check the "doc" folder.

The included EXAMPLE.CPP is headed by a bit test. Then, the following example is run:)

This is a simple example of quantum mechanics simulation in quantum computational logic. It is essentially a unidirectional binary quantum random walk algorithm, from a positive starting point, heading toward zero.

We assume a fixed length time step. During each time step, we step through an equal superposition of either standing still or taking one fixed length step from our current position toward our fixed destination.

This is equivalent to a physical body having a 50% chance of emitting a fixed unit of energy per a fixed unit of time, in a pure quantum state. Hence, it might be considered a simple quantum mechanics simulation.
