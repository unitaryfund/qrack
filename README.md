Copyright (c) Daniel Strano 2017. All rights reserved. (See "par_for.hpp" for additional information.)
Licensed under the GNU General Public License V3, (except where noted).
See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html for details.

This is a header-only, quick-and-dirty, multithreaded, universal quantum register simulation, allowing (nonphysical) register cloning and direct measurement of probability and phase, to leverage what advantages classical emulation of qubits can have.

To use:
1)#include "qrack.hpp"
2)Link against math and pthreads. (See build.sh for example.)

Instantiate a Qrack::Register, specifying the desired number of qubits. (Optionally, also specify the initial bit state in the constructor.)


PURE QUANTUM LOGIC:

Programs can be carried out in quantum computational logic on any register. See an online encyclopedia or a reference text for the function of the gates:

CCNOT(unsigned int control1, unsigned int control2, unsigned int target) - Doubly-controlled not

CNOT(unsigned int control, unsigned int target) - Controlled not

H(unsigned int bitIndex) - Hadamard

M(unsigned int bitIndex) - Measure (for bit = 1)

MAll(unsigned int permutation) - Measure for overall bit permutation state of register. (Permutations are numbered from 0 to 2^[number of qubits in register].)

R1(double radians, unsigned int bitIndex) - Rotate around |1> state

R1Dyad(int numerator, int denominator, unsigned int bitIndex) - Rotate around |1> state by an angle as a dyadic fraction, M_PI * numerator / denominator

RX(double radians, unsigned int bitIndex) - Rotate around x axis

RXDyad(int numerator, int denominator, unsigned int bitIndex) - Rotate around x axis by an angle as a dyadic fraction, M_PI * numerator / denominator

RY(double radians, unsigned int bitIndex) - Rotate around y axis

RYDyad(int numerator, int denominator, unsigned int bitIndex)- Rotate around y axis by an angle as a dyadic fraction, M_PI * numerator / denominator

RZ(double radians, unsigned int bitIndex) - Rotate around z axis

RZDyad(int numerator, int denominator, unsigned int bitIndex) - Rotate around z axis by an angle as a dyadic fraction, M_PI * numerator / denominator

Swap(unsigned int bitIndex1, unsigned int bitIndex2) - Swap the values of bits at the two bit indices

X(unsigned int bitIndex) - Apply Pauli x matrix

Y(unsigned int bitIndex) - Apply Pauli y matrix

Z(unsigned int bitIndex) - Apply Pauli z matrix


PSEUDO QUANTUM LOGIC:

It is not possible to perform even certain trivial classical operations directly on a quantum computer, like simply copying the state of a register to another register! However, since we are only emulating a quantum computer, we want to be able to write in quantum logic, while leveraging the abilities of a classical computer to our (speed) advantage. (Complex types are just aliases for standard C++ double accuracy complex number types. See header for details.)

Register(Register orig) - Copy constructor that clones the exact state of the emulated register

CloneRawState(Complex16 output[registerBitLength]) - Output the raw state of the emulated register, including phase information

Prob(unsigned int bitIndex) - Get the probability that the bit is in the |1> state, with maximum accuracy in one operation

ProbAll(unsigned int permutation) - Get the probability that the entire register is in the given permutation state, with maximum accuracy in one operation. (Permutations are numbered from 0 to 2^[number of qubits in register].)


OTHER METHODS:

unsigned int GetQubitCount() - Get the size of the register, in bits.

double Rand() - Generate a pseudo-random double, uniformly distributed from 0 to 1


EXAMPLE.CPP:

This is a simple example of quantum mechanics simulation in quantum computational logic. It is essentially a unidirectional binary quantum random walk algorithm, from a positive starting point, heading toward zero.

We assume a fixed length time step. During each time step, we step through an equal superposition of either standing still or taking one fixed length step from our current position toward our fixed destination.

This is equivalent to a physical body having a 50% chance of emitting a fixed unit of energy per a fixed unit of time, in a pure quantum state. Hence, it might be considered a simple quantum mechanics simulation.
