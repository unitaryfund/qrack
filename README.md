Copyright (c) Daniel Strano 2017. All rights reserved. (See "par_for.hpp" for additional information.)
Licensed under the GNU General Public License V3.
See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html for details.

This is a header-only, multithreaded framework for developing universal quantum processors, allowing (nonphysical) register cloning and direct measurement of probability and phase, to leverage what advantages classical emulation of qubits can have.

The intent of "Qrack" is to provide a framework for developing classically emulated universal quantum virtual machines. Theoretically, the virtual machines created with Qrack, and the code that runs on them, need not be packaged with each other. Qrack can be used to create a virtual machine that translates opcodes into instructions for a virtual quantum processor. Then, software that runs on the virtual machine could be abstracted from the machine itself as any higher-level software could. All that is necessary to run the quantum binary is any virtual machine with the same instruction set.

To use:

1)In your source code:
#include "qrack.hpp"
2)On the command line, in the project directory
make all

Make will link against math, pthreads, and OpenCL. Modify the makefile as necessary for inclusion in your project. Set the makefile QRACKVER variable to "qrack_ocl.cpp" for the OpenCL version, "qrack.cpp" for CPU parallelism only (without OpenCL), or "qrack_serial.cpp" for a totally non-parallel version. If you aren't using the OpenCL version, you should remove OpenCL linkage from the makefile. 

Instantiate a Qrack::CoherentUnit, specifying the desired number of qubits. (Optionally, also specify the initial bit state in the constructor.) Coherent units can be "cohered" and "decohered" with each other, to simulate coherence and loss of coherence between distinct quantum bit registers. Both single quantum gate commands and register-like multi-bit commands are available.

If you are using the OpenCL version of the headers, the OpenCL platform and device can be selected in the first call to the OpenCL context instance. (See "example.cpp" for details. The OpenCL context is managed by a singleton, shared between an arbitrary number of CoherentUnit objects.)

For more information, compile the doxygen.config in the root folder, and then check the "doc" folder.

EXAMPLE.CPP:

(The example is headed with several register operation tests. Then, the following example is run:)

This is a simple example of quantum mechanics simulation in quantum computational logic. It is essentially a unidirectional binary quantum random walk algorithm, from a positive starting point, heading toward zero.

We assume a fixed length time step. During each time step, we step through an equal superposition of either standing still or taking one fixed length step from our current position toward our fixed destination.

This is equivalent to a physical body having a 50% chance of emitting a fixed unit of energy per a fixed unit of time, in a pure quantum state. Hence, it might be considered a simple quantum mechanics simulation.
