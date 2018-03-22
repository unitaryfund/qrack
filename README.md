# Qrack

[![Quantum Rack Build Status](https://api.travis-ci.org/vm6502q/qrack.svg?branch=master)](https://travis-ci.org/vm6502q/qrack/builds)

This is a multithreaded framework for developing classically emulated virtual
universal quantum processors. (See the doxygen entry for "CoherentUnit" for an
outline of the algorithms by which Qrack is implemented.)

The intent of "Qrack" is to provide a framework for developing classically
emulated universal quantum virtual machines. In addition to quantum gates,
Qrack provides optimized versions of multi-bit, register-wise, opcode-like
"instructions." A chip-like quantum CPU (QCPU) is instantiated as a
"Qrack::CoherentUnit," assuming all the memory in the quantum memory in the
QCPU is quantum mechanically "coherent" for quantum computation.

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

## Overview of Algorithms

The Doxygen reference in this project and this README serve two primary purposes. Secondarily, the Doxygen contains a quick reference for the "Qrack" API. Primarily, it is meant to teach enough about practical quantum computation and emulation that one can understand the implementation of all methods in Qrack to the point of being able to reimplement the algorithms equivalently on one's own. This section contains an overview of the general method whereby Qrack implements basically all of its gate and register functionality.

Like classical bits, a set of qubits has a maximal respresentation as the permutation of bits. (An 8 bit byte has 256 permutations, commonly numbered 0 to 255, as does an 8 bit qubyte.) Additionally, the state of a qubyte is fully specified in terms of probability and phase of each permutation of qubits. This is the "|0>/|1>" "permutation basis." There are other fully descriptive bases, such as the |+>/|-> permutation basis, which is characteristic of Hadamard gates. The notation "|x>" represents a "ket" of the "x" state in the quantum "bra-ket" notation of Dirac. It is a quantum state vector as described by Schrödinger's equation. When we say |01>, we mean the qubit equivalent of the binary bit pemutation "01."

The state of a two bit permutation can be described as follows: where one in the set of variables "x_0, x_1, x_2, and x_3" is equal to 1 and the rest are equal to zero, the state of the bit permutation can always be described by

|psi> = x_0 * |00> + x_1 * |01> + x_2 * |10> + x_3 * |11>.

One of the leading variables is always 1 and the rest are always 0. That is, the state of the classical bit combination is always exactly one of |00>, |01>, |10>, or |11>, and never a mix of them at once, however we would mix them. One way to mix them is probabilistically, in which the sum of probabilities of states should be 100% or 1. For example, this suggests splitting x_0 and x_1 into 1/2 and 1/2 to represent a potential |psi>, but Schrödinger's equation actually requires us to split into 1/sqrt(2) and 1/sqrt(2) to get 100% probability, like so,

|psi> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |10>,

where the leading coefficients are ultimately squared to give probabilities. This is a valid description of a 2 qubit permutation. The first equation given before it above encompasses all possible states of a 2 qubit combination, when the x_n variables are constrained so that the total probability of all states adds up to one. However, the domain of the x_n variables must also be the complex numbers. This is also a valid state, for example:

|psi> = (1+i)/ (2 * sqrt(2)) * |00> + (1-i) / (2 * sqrt(2)) * |10>

where "i" is defined as the sqrt(-1). This imparts "phase" to each permutation state vector component like |00> or |10>, (which are "eigenstates"). Phase and probability of permutation state fully (but not uniquely) specify the state of a coherent set of qubits.

For N bits, there are 2^N permutation basis "eigenstates" that with probability normalization and phase fully describe every possible quantum state of the N qubits. A CoherentUnit tracks the 2^N dimensional state vector of eigenstate components, each permutation carrying probability and phase. It optimizes certain register-like methods by operating in parallel over the "entanglements" of these permutation basis states. For example, the state

|psi> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |11>

has a probablity of both bits being 1 or neither bit being 1, but it has no independent probability for the bits being different, when measured. If this state is acted on by an X or NOT gate on the left qubit, for example, we need only act on the states entangled into the original state:

|psi_0> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |11>
(When acted on by an X gate on the left bit, goes to:)
|psi_1> = 1 / sqrt(2) * |10> + 1 / sqrt(2) * |01>

In the permutation basis, "entanglement" is as simple as the ability to restrain bit combinations in specificying an arbitrary "|psi>" state, as we have just described at length.

In Qrack, simple gates are represented by small complex number matrices, generally 2x2 components, that act on pairings of state vector components with the target qubit being 0 or 1 and all other qubits being held fixed in a loop iteration. For example, in an 8 qubit system, acting a single bit gate on the leftmost qubit, these two states become paired:

|00101111>
and
|10101111>.

Similarly, these states also become paired:

|00101100>
and
|10101100>,

And so on for all states in which the seven uninvolved bits are kept the same, but 0 and 1 states are paired for the bit acted on by the gate. This covers the entire permutation basis, a full description of all possible quantum states of the CoherentUnit, with pairs of two state vector components acted on by a 2x2 matrix. For example, for the Z gate, acting it on a single bit is equivalent to multiplying a single bit state vector by this matrix:

[  1  0 ]
[  0 -1 ] (is a Z gate)

The single qubit state vector has two components:

[ x_0 ]
[ x_1 ] (represents the permutations of a single qubit).

These "x_0" and "x_1" are the same type of coefficients described above,

|psi> = x_0 * |0> + x_1 * |1>

and the action of a gate is a matrix multiplication:

[  1  0 ] * [ x_0 ] = [ x_0 ]
[  0 -1 ]   [ x_1 ]   [-x_1 ].

For 2 qubits, we can form 4x4 matrices to act on 4 permutation eigenstates. For 3 qubits, we can form 8x8 matrices to act on 8 permutation eigenstates, and so on. However, for gates acting on single bits in states with large numbers of qubits, it is actually not necessary to carry out any matrix multiplication larger than a 2x2 matrix acting acting on a sub-state vector of 2 components. Again, we pair all permutation state vector components where all qubits are the same same, except for the one bit being acted on, for which we pair 0 and 1. Again, for example, acting on the leftmost qubit,

|00100011>
is paired with
|10100011>,

and
|00101011>
is paired with
|10101011>,

and
|01101011>
is paired with
|11101011>,

and we can carry out the gate in terms of only 2x2 complex number matrix multiplications, which is a massive optimization and "embarrassingly parallel." (Further, Qrack already employs POSIX thread type parallelism, SIMD parallelism for complex number operations, and kernel-type GPU parallelism.)

For register-like operations, we can optimize beyond this level for single bit gates. If a virtual quantum chip has multiple registers that can be entangled, by requirements of the minimum full physical description of a quantum mechanical state, the registers must usually be all contained in a single CoherentUnit. So, for 2 8 bit registers, we might have one 16 bit CoherentUnit. For a bitwise NOT or X operation on one register, we can take an initial entangled state and sieve out initial register states to be mapped to final register states. For example, say we start with an entangled state:

|psi> = 1/sqrt(2) * |01010101 11111110> - 1/sqrt(2) |10101010 00000000>.

The registers are "entangled" so that only two possible states can result from measurement; if we measure any single bit, (except the right-most, in this example,) we collapse into one of these two states, adjusting the normalization so that only state remains in the full description of the quantum state.. (In general, measuring a single bit might only partially collapse the entanglement, as more than one state could potentially be consistent with the same qubit measurement outcome as 0 or 1. This is the case for the right-most bit; measuring it from this example initial state will always yield "0" and tell us nothing else about the overall permutation state, leaving the state uncollapsed. Measuring any bit except the right-most will collapse the entire set of bits into a single permutation.)

Say we want to act a bitwise NOT or X operation on the right-hand register of 8 bits. We simply act the NOT operation simultaneously on all of the right-hand bits in all entangled input states:

|psi_0> = 1/sqrt(2) * |01010101 11111110> - 1/sqrt(2) |10101010 00000000>
(acted on by a bitwise NOT or X on the right-hand 8 bit register becomes)
|psi_1> = 1/sqrt(2) * |01010101 00000001> - 1/sqrt(2) |10101010 11111111>

This is again "embarrassingly parallel." Some bits are completely uninvolved, (the left-hand 8 bits, in this case,) and these bits are passed unchanged in each state from input to output. Bits acted on by the register operation have a one-to-one mapping between input and states. This can all be handled via transformation via bit masks on the input state permutation index. And, in fact, bits are not rearranged in the state vector at all; it is the "x_n" complex number coefficients which are rearranged according to this bitmask transformation and mapping of the input state to the output state! (The coefficient "x_i" of state |01010101 11111110> is switched for the coefficient "x_j" of state |01010101 00000001>, and only the coefficients are rearranged, with a mapping that's determined via bitmask transformations.) This is almost the entire principle behind the algorithms for optimized register-like methods in Qrack. See also the register-wise "CoherentUnit::X" gate implementation in "qregister.cpp" for inline documentation on this general algorithm by which basically all register-wise gates operate.

Quantum gates are represented by "unitary" matrices. Unitary matrices preserve the norm (length) of state vectors. Quantum physically observable quantities are associated with "Hermitian" unitary matrices, which are equal to their own conjugate transpose. Not all gates are Hermitian or associated with quantum observables, like general rotation operators. (Three dimensions of spin can be physically measured; the act of rotating spin along these axes is not associated with independent measurable quantities.) The Qrack project is targeted to efficient and practical classical emulation of ideal, noiseless systems of qubits, and so does not concern itself with hardware noise, error correction, or restraining emulation to gates which have already been realized in physical hardware. If a hypothetical gate is at least unitary, and if it is logically expedient for quantum emulation, the design intent of Qrack permits it as a method in the API.

The act of measuring a bit "collapses" its quantum state in the sense of breaking unitary evolution of state. See the doxygen for the M() method for a discussion of measurement and unitarity.

Additionally, as Qrack targets classical emulation of quantum hardware, certain convenience methods can be employed in classical emulation which are not physically or practically attainable in quantum hardware, such as the "cloning" of arbitrary pure quantum states and the direct nondestructive measurement of probability and phase. Members of this limited set of convenience methods are marked "PSEUDO-QUANTUM" in the API reference and need not be employed at all.

## Note about unitarity and arithmetic

The project's author notes that the ADD and SUB variants in the current version of the project break unitarity and are not on rigorous footing. They are included in the project as the basis for work on correct implementations. INC and DEC, however, are unitary and function much like SWAP operations. ADD and SUB operations are provisional and will be corrected.

Similarly, AND/OR/XOR are only provided for convenience and generally entail a measurement of the output bit. For register-based virtual quantum processors, we suspect it will be a common requirement that an output register be measured, cleared, and loaded with the output logical comparison operations, but this will generally require measurement and therefore introduce a random phase factor. CCNOT and X gates (composed for convenience as "AntiCCNOT" gates) could instead be operated on a target bit with a known input state to achieve a similar result in a unitary fashion, but this is left to the particular virtual machine implementation.

Similarly, the "Decohere" and "Dispose" methods should only be used on qubits that are guaranteed to be separable.

Qrack is an experimental work in progress, and the author aims for both utility and correctness, but the project cannot be guaranteed to be fit for any purpose, express or implied. (See LICENSE.md for details.)

## Copyright and License

Copyright (c) Daniel Strano 2017, (with many thanks to Benn Bollay for tool
chain development in particular, and also Marek Karcz for supplying an awesome
base classical 6502 emulator for proof-of-concept). All rights reserved. (See
"par_for.hpp" for additional information.)

Licensed under the GNU General Public License V3.

See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html for details.


