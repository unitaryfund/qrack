//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>
#include <atomic>
#include <ctime>
#include <future>
#include <math.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <thread>

#include "complex16simd.hpp"

#define Complex16 Complex16Simd
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack {

class CoherentUnit;

/*
 * Enumerated list of supported engines.
 *
 * Not currently published since selection isn't supported by the API.
 */
enum CoherentUnitEngine {
    COHERENT_UNIT_ENGINE_SOFTWARE = 0,
    COHERENT_UNIT_ENGINE_OPENCL,

    COHERENT_UNIT_ENGINE_MAX
};

CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState);

/// The "Qrack::CoherentUnit" class represents one or more coherent quantum processor registers, including primitive bit
/// logic gates and (abstract) opcodes-like methods.
/**
 * A "Qrack::CoherentUnit" is a qubit permutation state vector with methods to operate on it as by gates and
register-like instructions. In brief: All directly interacting qubits must be contained in a single CoherentUnit object,
by requirement of quantum mechanics, unless a certain collection of bits represents a "separable quantum subsystem." All
registers of a virtual chip will usually be contained in a single CoherentUnit, and they are accesible similar to a
one-dimensional array of qubits.

Introduction: The Doxygen reference in this project serves two primary purposes. Secondarily, it is a quick reference
for the "Qrack" API. Primarily, it is meant to teach enough about practical quantum computation and emulation that one
can understand the implementation of all methods in Qrack to the point of being able to reimplement the algorithms
equivalently on one's own. This entry contains an overview of the general method whereby Qrack implements basically all
of its gate and register functionality.

Like classical bits, a set of qubits has a maximal respresentation as the permutation of bits. (An 8 bit
byte has 256 permutations, commonly numbered 0 to 255, as does an 8 bit qubyte.) Additionally, the state of a qubyte is
fully specified in terms of probability and phase of each permutation of qubits. This is the "|0>/|1>" "permutation
basis." There are other fully descriptive bases, such as the |+>/|-> permutation basis, which is characteristic of
Hadamard gates. The notation "|x>" represents a "ket" of the "x" state in the quantum "bra-ket" notation of Dirac. It is
a quantum state vector as described by Schrödinger's equation. When we say |01>, we mean the qubit equivalent of the
binary bit pemutation "01."

The state of a two bit permutation can be described as follows: where one in the set of variables "x_0, x_1, x_2, and
x_3" is equal to 1 and the rest are equal to zero, the state of the bit permutation can always be described by

|psi> = x_0 * |00> + x_1 * |01> + x_2 * |10> + x_3 * |11>

One of the leading variables is always 1 and the rest are always 0. That is, the state of the classical bit combination
is always exactly one of |00>, |01>, |10>, or |11>, and never a mix of them at once, however we would mix them. One way
to mix them is probabilistically, in which the sum of probabilities of states should be 100% or 1. For example, this
suggests splitting x_0 and x_1 into 1/2 and 1/2 to represent a potential |psi>, but Schrödinger's equation actually
requires us to split into 1/sqrt(2) and 1/sqrt(2) to get 100% probability, like so,

|psi> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |10>,

where the leading coefficients are ultimately squared to give probabilities. This is a valid description of a 2 qubit
permutation. The first equation given before it above encompasses all possible states of a 2 qubit combination, when the
x_n variables are constrained so that the total probability of all states adds up to one. However, the domain of the x_n
variables must also be the complex numbers. This is also a valid state, for example:

|psi> = (1+i)/ (2 * sqrt(2)) * |00> + (1-i) / (2 * sqrt(2)) * |10>


where "i" is defined as the sqrt(-1). This imparts "phase" to each permutation state vector component like |00> or |10>,
(which are "eigenstates"). Phase and probability of permutation state fully (but not uniquely) specify the state of a
coherent set of qubits.

For N bits, there are 2^N permutation basis "eigenstates" that with probability normalization and phase fully describe
every possible quantum state of the N qubits. A CoherentUnit tracks the 2^N dimensional state vector of eigenstate
components, each permutation carrying probability and phase. It optimizes certain register-like methods by operating in
parallel over the "entanglements" of these permutation basis states. For example, the state

|psi> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |11>

has a probablity of both bits being 1 or neither bit being 1, but it has no independent probability for the bits being
different, when measured. If this state is acted on by an X or NOT gate on the left qubit, for example, we need only act
on the states entangled into the original state:

|psi_0> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |11> <br/>
(When acted on by an X gate on the left bit, goes to:) <br/>
|psi_1> = 1 / sqrt(2) * |10> + 1 / sqrt(2) * |01> <br/>

In the permutation basis, "entanglement" is as simple as the ability to restrain bit combinations in specificying an
arbitrary "|psi>" state, as we have just described at length.

In Qrack, simple gates are represented by small complex number matrices, generally 2x2 components, that act on pairings
of state vector components with the target qubit being 0 or 1 and all other qubits being held fixed in a loop iteration.
For example, in an 8 qubit system, acting a single bit gate on the leftmost qubit, these two states become paired:

|00101111> <br/>
and <br/>
|10101111>. <br/>

Similarly, these states also become paired:

|00101100>
and
|10101100>,

And so on for all states in which the seven uninvolved bits are kept the same, but 0 and 1 states are paired for the bit
acted on by the gate. This covers the entire permutation basis, a full description of all possible quantum states of the
CoherentUnit, with pairs of two state vector components acted on by a 2x2 matrix. For example, for the Z gate, acting it
on a single bit is equivalent to multiplying a single bit state vector by this matrix:

[  1  0 ] <br/>
[  0 -1 ] (is a Z gate)

The single qubit state vector has two components:

[ x_0 ] <br/>
[ x_1 ] (represents the permutations of a single qubit).

These "x_0" and "x_1" are the same type of coefficients described above,

|psi> = x_0 * |0> + x_1 * |1>

and the action of a gate is a matrix multiplication:

[  1  0 ] *      [ x_0 ] =       [ x_0 ] <br/>
[  0 -1 ] &nbsp; [ x_1 ] &nbsp;  [-x_1 ].

For 2 qubits, we can form 4x4 matrices to act on 4 permutation eigenstates. For 3 qubits, we can form 8x8 matrices to
act on 8 permutation eigenstates, and so on. However, for gates acting on single bits in states with large numbers of
qubits, it is actually not necessary to carry out any matrix multiplication larger than a 2x2 matrix acting acting on a
sub-state vector of 2 components. Again, we pair all permutation state vector components where all qubits are the same
same, except for the one bit being acted on, for which we pair 0 and 1. Again, for example, acting on the leftmost
qubit,

|00100011><br/>
is paired with<br/>
|10100011>,<br/>

and<br/>
|00101011><br/>
is paired with<br/>
|10101011>,<br/>

and<br/>
|01101011><br/>
is paired with<br/>
|11101011>,<br/>

and we can carry out the gate in terms of only 2x2 complex number matrix multiplications, which is a massive
optimization and "embarrassingly parallel." (Further, Qrack already employs POSIX thread type parallelism, SIMD
parallelism for complex number operations, and kernel-type GPU parallelism.)

For register-like operations, we can optimize beyond this level for single bit gates. If a virtual quantum chip has
multiple registers that can be entangled, by requirements of the minimum full physical description of a quantum
mechanical state, the registers must usually be all contained in a single CoherentUnit. So, for 2 8 bit registers, we
might have one 16 bit CoherentUnit. For a bitwise NOT or X operation on one register, we can take an initial
entangled state and sieve out initial register states to be mapped to final register states. For example, say we start
with an entangled state:

|psi> = 1/sqrt(2) * |01010101 11111110> - 1/sqrt(2) |10101010 00000000>.

The registers are "entangled" so that only two possible states can result from measurement; if we measure any single
bit, (except the right-most, in this example,) we collapse into one of these two states, adjusting the normalization so
that only state remains in the full description of the quantum state.. (In general, measuring a single bit might only
partially collapse the entanglement, as more than one state could potentially be consistent with the same qubit
measurement outcome as 0 or 1. This is the case for the right-most bit; measuring it from this example initial state
will always yield "0" and tell us nothing else about the overall permutation state, leaving the state uncollapsed.
Measuring any bit except the right-most will collapse the entire set of bits into a single permutation.)

Say we want to act a bitwise NOT or X operation on the right-hand register of 8 bits. We simply act the NOT operation
simultaneously on all of the right-hand bits in all entangled input states:

|psi_0> = 1/sqrt(2) * |01010101 11111110> - 1/sqrt(2) |10101010 00000000>
<br/>(acted on by a bitwise NOT or X on the right-hand 8 bit register becomes)<br/>
|psi_1> = 1/sqrt(2) * |01010101 00000001> - 1/sqrt(2) |10101010 11111111>

This is again "embarrassingly parallel." Some bits are completely uninvolved, (the left-hand 8 bits, in this case,) and
these bits are passed unchanged in each state from input to output. Bits acted on by the register operation have a
one-to-one mapping between input and states. This can all be handled via transformation via bit masks on the input state
permutation index. And, in fact, bits are not rearranged in the state vector at all; it is the "x_n" complex number
coefficients which are rearranged according to this bitmask transformation and mapping of the input state to the output
state! (The coefficient "x_i" of state |01010101 11111110> is switched for the coefficient "x_j" of state |01010101
00000001>, and only the coefficients are rearranged, with a mapping that's determined via bitmask transformations.)
This is almost the entire principle behind the algorithms for optimized register-like methods in Qrack.

Quantum gates are represented by "unitary" matrices. Unitary matrices preserve the norm (length) of state vectors.
Quantum physically observable quantities are associated with "Hermitian" unitary matrices, which are equal to their own
conjugate transpose. Not all gates are Hermitian or associated with quantum observables, like general rotation
operators. (Three dimensions of spin can be physically measured; the act of rotating spin along these axes is not
associated with independent measurable quantities.) The Qrack project is targeted to efficient and practical classical
emulation of ideal, noiseless systems of qubits, and so does not concern itself with hardware noise, error correction,
or restraining emulation to gates which have already been realized in physical hardware. If a hypothetical gate is at
least unitary, and if it is logically expedient for quantum emulation, the design intent of Qrack permits it as a method
in the API.

The act of measuring a bit "collapses" its quantum state in the sense of breaking unitary evolution of state. See M()
for a discussion of measurement and unitarity.

Additionally, as Qrack targets classical emulation of quantum hardware, certain convenience methods can be employed in
classical emulation which are not physically or practically attainable in quantum hardware, such as the "cloning" of
arbitrary pure quantum states and the direct nondestructive measurement of probability and phase. Members of this
limited set of convenience methods are marked "PSEUDO-QUANTUM" in the API reference and need not be employed at all.

 */
class CoherentUnit {
public:
    /// Initialize a coherent unit with qBitCount number of bits, all to |0> state.
    CoherentUnit(bitLenInt qBitCount);

    /// Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state
    CoherentUnit(bitLenInt qBitCount, bitCapInt initState);

    /// PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
    CoherentUnit(const CoherentUnit& pqs);

    /// Destructor of CoherentUnit. (Permutation state vector is declared on heap, as well as some OpenCL objects.)
    virtual ~CoherentUnit() {}

    /// Set the random seed (primarily used for testing)
    void SetRandomSeed(uint32_t seed);

    /// Get the count of bits in this register
    int GetQubitCount() { return qubitCount; }

    /// Get the 1 << GetQubitCount()
    int GetMaxQPower() { return maxQPower; }

    /// PSEUDO-QUANTUM Output the exact quantum state of this register as a permutation basis array of complex numbers
    void CloneRawState(Complex16* output);

    /// Generate a random double from 0 to 1
    double Rand();

    /// Set |0>/|1> bit basis pure quantum permutation state, as an unsigned int
    void SetPermutation(bitCapInt perm);

    /// Set arbitrary pure quantum state, in unsigned int permutation basis
    void SetQuantumState(Complex16* inputState);

    /// Combine (a copy of) another CoherentUnit with this one, after the last bit index of this one.
    /**
     * "Cohere" is combines the quantum description of state of two independent CoherentUnit objects into one object,
containing the full permutation basis of the full object. The "inputState" bits are added after the last qubit index of
the CoherentUnit to which we "Cohere." Informally, "Cohere" is equivalent to "just setting another group of qubits down
next to the first" without interacting them. Schroedinger's equation can form a description of state for two independent
subsystems at once or "separable quantum subsystems" without interacting them. Once the description of state of the
independent systems is combined, we can interact them, and we can describe their entanglements to each other, in which
case they are no longer independent. A full entangled description of quantum state is not possible for two independent
quantum subsystems until we "Cohere" them.

"Cohere" multiplies the probabilities of the indepedent permutation states of the two subsystems to find the
probabilites of the entire set of combined permutations, by simple combinatorial reasoning. If the probablity of the
"left-hand" subsystem being in |00> is 1/4, and the probablity of the "right-hand" subsystem being in |101> is 1/8, than
the probability of the combined |00101> permutation state is 1/32, and so on for all permutations of the new combined
state.

If the programmer doesn't want to "cheat" quantum mechanically, then the original copy of the state which is duplicated
into the larger CoherentUnit should be "thrown away" to satisfy "no clone theorem." This is not semantically enforced in
Qrack, because optimization of an emulator might be acheived by "cloning" "under-the-hood" while only exposing a quantum
mechanically consistent API or instruction set.
     */
    void Cohere(CoherentUnit& toCopy);

    /// Minimally decohere a set of contiguous bits from the full coherent unit, into "destination."
    /**
     * Minimally decohere a set of contigious bits from the full coherent unit. The length of this coherent unit is
reduced by the length of bits decohered, and the bits removed are output in the destination CoherentUnit pointer. The
destination object must be initialized to the correct number of bits, in 0 permutation state. For quantum mechanical
accuracy, the bit set removed and the bit set left behind should be quantum mechanically "separable."

Like how "Cohere" is like "just setting another group of qubits down next to the first," <b><i>if two sets of qubits are
not entangled,</i></b> then "Decohere" is like "just moving a few qubits away from the rest." Schroedinger's equation
does not require bits to be explicitly interacted in order to describe their permutation basis, and the descriptions of
state of <b>separable</b> subsystems, those which are not entangled with other subsystems, are just as easily removed
from the description of state.

If we have for example 5 qubits, and we wish to separate into "left" and "right" subsystems of 3 and 2 qubits, we sum
probabilities of one permutation of the "left" three over ALL permutations of the "right" two, for all permutations, and
vice versa, like so:

prob(|(left) 1000>) = prob(|1000 00>) + prob(|1000 10>) + prob(|1000 01>) + prob(|1000 11>).

If the subsystems are not "separable," i.e. if they are entangled, this operation is not well-motivated, and its output
is not necessarily defined. (The summing of probabilities over permutations of subsytems will be performed as described
above, but this is not quantum mechanically meaningful.) To ensure that the subsystem is "separable," i.e. that it has
no entanglements to other subsystems in the CoherentUnit, it can be measured with M(), or else all qubits <i>other
than</i> the subsystem can be measured.
     */
    void Decohere(bitLenInt start, bitLenInt length, CoherentUnit& destination);

    /// Minimally decohere a set of contigious bits from the full coherent unit, throwing these qubits away.
    /**
     * Minimally decohere a set of contigious bits from the full coherent unit, discarding these bits. The length of
this coherent unit is reduced by the length of bits decohered. For quantum mechanical accuracy, the bit set removed and
the bit set left behind should be quantum mechanically "separable."

Like how "Cohere" is like "just setting another group of qubits down next to the first," <b><i>if two sets of qubits are
not entangled,</i></b> then "Dispose" is like "just moving a few qubits away from the rest, and throwing them in the
trash." Schroedinger's equation does not require bits to be explicitly interacted in order to describe their permutation
basis, and the descriptions of state of <b>separable</b> subsystems, those which are not entangled with other
subsystems, are just as easily removed from the description of state.

If we have for example 5 qubits, and we wish to separate into "left" and "right" subsystems of 3 and 2 qubits, we sum
probabilities of one permutation of the "left" three over ALL permutations of the "right" two, for all permutations, and
vice versa, like so:

prob(|(left) 1000>) = prob(|1000 00>) + prob(|1000 10>) + prob(|1000 01>) + prob(|1000 11>).

If the subsystems are not "separable," i.e. if they are entangled, this operation is not well-motivated, and its output
is not necessarily defined. (The summing of probabilities over permutations of subsytems will be performed as described
above, but this is not quantum mechanically meaningful.) To ensure that the subsystem is "separable," i.e. that it has
no entanglements to other subsystems in the CoherentUnit, it can be measured with M(), or else all qubits <i>other
than</i> the subsystem can be measured.
     */
    void Dispose(bitLenInt start, bitLenInt length);

    // Logic Gates
    //
    // Each bit is paired with a CL* variant that utilizes a classical bit as
    // an input.

    /// Quantum analog of classical "AND" gate. Measures the outputBit, then overwrites it with result.
    void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    /// Quantum analog of classical "AND" gate. Takes one qubit input and one classical bit input. Measures the
    /// outputBit, then overwrites it with result.
    void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /// Quantum analog of classical "OR" gate. Measures the outputBit, then overwrites it with result.
    void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    /// Quantum analog of classical "AND" gate. Takes one qubit input and one classical bit input. Measures the
    /// outputBit, then overwrites it with result.
    void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /// Quantum analog of classical "exclusive-OR" or "XOR" gate. Measures the outputBit, then overwrites it with
    /// result.
    void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    /// Quantum analog of classical "exclusive-OR" or "XOR" gate. Takes one qubit input and one classical bit input.
    /// Measures the outputBit, then overwrites it with result.
    void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /// "Doubly-controlled NOT gate." If both controls are set to 1, the target bit is NOT-ed or X-ed.
    void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    /// "Anti doubly-controlled NOT gate." If both controls are set to 0, the target bit is NOT-ed or X-ed.
    void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);

    /// "Controlled NOT gate." If the control is set to 1, the target bit is NOT-ed or X-ed.
    void CNOT(bitLenInt control, bitLenInt target);
    /// "Anti controlled NOT gate." If the control is set to 0, the target bit is NOT-ed or X-ed.
    void AntiCNOT(bitLenInt control, bitLenInt target);

    /// Hadamard gate. Applies a Hadamard gate on qubit at "qubitIndex."
    void H(bitLenInt qubitIndex);
    /// "Measurement gate." Measures the qubit at "qubitIndex" and returns either "true" or "false." (This "gate" breaks
    /// unitarity.)
    /**
        All physical evolution of a quantum state should be "unitary," except measurement. Measurement of a qubit
"collapses" the quantum state into either only permutation states consistent with a |0> state for the bit, or else only
permutation states consistent with a |1> state for the bit. Measurement also effectively multiplies the overall quantum
state vector of the system by a random phase factor, equiprobable over all possible phase angles.

Effectively, when a bit measurement is emulated, Qrack calculates the norm of all permutation state components, to find
their respective probabilities. The probabilities of all states in which the measured bit is "0" can be summed to give
the probability of the bit being "0," and separately the probabilities of all states in which the measured bit is "1"
can be summed to give the probability of the bit being "1." To simulate measurement, a random float between 0 and 1 is
compared to the sum of the probability of all permutation states in which the bit is equal to "1". Depending on whether
the random float is higher or lower than the probability, the qubit is determined to be either |0> or |1>, (up to
phase). If the bit is determined to be |1>, then all permutation eigenstates in which the bit would be equal to |0> have
their probability set to zero, and vice versa if the bit is determined to be |0>. Then, all remaining permutation states
with nonzero probability are linearly rescaled so that the total probability of all permutation states is again
"normalized" to exactly 100% or 1, (within double precision rounding error). Physically, the act of measurement should
introduce an overall random phase factor on the state vector, which is emulated by generating another constantly
distributed random float to select a phase angle between 0 and 2 * Pi.

Measurement breaks unitary evolution of state. All quantum gates except measurement should generally act as a unitary
matrix on a permutation state vector. (Note that Boolean comparison convenience methods in Qrack such as "AND," "OR,"
and "XOR" employ the measurement operation in the act of first clearing output bits before filling them with the result
of comparison, and these convenience methods therefore break unitary evolution of state, but in a physically realistic
way. Comparable unitary operations would be performed with a combination of X and CCNOT gates, also called "Toffoli"
gates, but the output bits would have to be assumed to be in a known fixed state, like all |0>, ahead of time to produce
unitary logical comparison operations.)
      */
    bool M(bitLenInt qubitIndex);

    /// "X gate." Applies the Pauli "X" operator to the qubit at "qubitIndex." The Pauli "X" operator is equivalent to a
    /// logical "NOT."
    void X(bitLenInt qubitIndex);
    /// "Y gate." Applies the Pauli "Y" operator to the qubit at "qubitIndex." The Pauli "Y" operator is similar to a
    /// logical "NOT" with permutation phase effects.
    void Y(bitLenInt qubitIndex);
    /// "Z gate." Applies the Pauli "Z" operator to the qubit at "qubitIndex." The Pauli "Z" operator reverses the phase
    /// of |1> and leaves |0> unchanged.
    void Z(bitLenInt qubitIndex);

    // Controlled variants
    /// "Controlled Y gate." If the "control" bit is set to 1, then the Pauli "Y" operator is applied to "target."
    void CY(bitLenInt control, bitLenInt target);
    /// "Controlled Z gate." If the "control" bit is set to 1, then the Pauli "Z" operator is applied to "target."
    void CZ(bitLenInt control, bitLenInt target);

    /// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
    double Prob(bitLenInt qubitIndex);

    /// PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
    double ProbAll(bitCapInt fullRegister);

    /// PSEUDO-QUANTUM Direct measure of all bit probabilities in register to be in |1> state
    void ProbArray(double* probArray);

    /*
     * Rotational gates:
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     */

    /// "Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state
    void R1(double radians, bitLenInt qubitIndex);

    /// Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / denominator) around |1> state.
    void R1Dyad(int numerator, int denominator, bitLenInt qubitIndex);

    /// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
    void RX(double radians, bitLenInt qubitIndex);
    /// Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis.
    void RXDyad(int numerator, int denominator, bitLenInt qubitIndex);
    /// Controlled x axis rotation gate - If "control" is set to 1, rotates as e^(-i*\theta/2) around Pauli x axis
    void CRX(double radians, bitLenInt control, bitLenInt target);
    /// Controlled dyadic fraction x axis rotation gate - If "control" is set to 1, rotates as e^(i*(M_PI * numerator) /
    /// denominator) around Pauli x axis.
    void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
    void RY(double radians, bitLenInt qubitIndex);
    /// Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis.
    void RYDyad(int numerator, int denominator, bitLenInt qubitIndex);
    /// Controlled y axis rotation gate - If "control" is set to 1, rotates as e^(-i*\theta/2) around Pauli y axis
    void CRY(double radians, bitLenInt control, bitLenInt target);
    /// Controlled dyadic fraction y axis rotation gate - If "control" is set to 1, rotates as e^(i*(M_PI * numerator) /
    /// denominator) around Pauli y axis.
    void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
    void RZ(double radians, bitLenInt qubitIndex);
    /// Dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z axis.
    void RZDyad(int numerator, int denominator, bitLenInt qubitIndex);
    /// Controlled z axis rotation gate - If "control" is set to 1, rotates as e^(-i*\theta/2) around Pauli z axis
    void CRZ(double radians, bitLenInt control, bitLenInt target);
    /// Controlled dyadic fraction z axis rotation gate - If "control" is set to 1, rotates as e^(i*(M_PI * numerator) /
    /// denominator) around Pauli z axis.
    void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /// Set individual bit to pure |0> (false) or |1> (true) state
    /**
     * To set a bit, the bit is first measured. If the result of measurement matches "value," the bit is considered set.
     * If the result of measurement is the opposite of "value," an X gate is applied to the bit. The state ends up
     * entirely in the "value" state, with a random phase factor.
     */
    void SetBit(bitLenInt qubitIndex1, bool value);

    /// Swap values of two bits in register
    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    /// Controlled "phase shift gate" - if control bit is set to 1, rotates target bit as e^(-i*\theta/2) around |1>
    /// state
    void CRT(double radians, bitLenInt control, bitLenInt target);
    /// Controlled dyadic fraction "phase shift gate" - if control bit is set to 1, rotates target bit as e^(i*(M_PI *
    /// numerator) / denominator) around |1> state
    void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    // Register-spanning gates
    //
    // Convienence functions implementing gates are applied from the bit
    // 'start' for 'length' bits for the register.

    /// Bitwise Hadamard
    void H(bitLenInt start, bitLenInt length);
    /// Bitwise Pauli X (or logical "NOT") operator
    void X(bitLenInt start, bitLenInt length);
    /// Bitwise Pauli Y operator
    void Y(bitLenInt start, bitLenInt length);
    /// Bitwise Pauli Z operator
    void Z(bitLenInt start, bitLenInt length);

    /// Bitwise controlled-not with "inputStart1" bits as controls and "inputStart2" bits as targets, with registers of
    /// "length" bits.
    void CNOT(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt length);
    /// Bitwise "AND" between registers at "inputStart1" and "inputStart2," with registers of "length" bits.
    void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    /// Bitwise "AND" between qubit register at "qInputStart" and the classical bits of "classicalInput," with registers
    /// of "length" bits.
    void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    /// Bitwise "OR" between registers at "inputStart1" and "inputStart2," with registers of "length" bits.
    void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    /// Bitwise "OR" between qubit register at "qInputStart" and the classical bits of "classicalInput," with registers
    /// of "length" bits.
    void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    /// Bitwise "exclusive OR" or "XOR" between registers at "inputStart1" and "inputStart2," with registers of "length"
    /// bits.
    void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    /// Bitwise "exclusive OR" or "XOR" between qubit register at "qInputStart" and the classical bits of
    /// "classicalInput," with registers of "length" bits.
    void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /// Bitwise |1> axis rotation gate - Rotates each bit from "start" for "length" as e^(-i*\theta/2) around the |1>
    /// state.
    void R1(double radians, bitLenInt start, bitLenInt length);
    /// Bitwise dyadic |1> axis rotation gate - Rotates each bit from "start" for "length" as e^(i*(M_PI * numerator) /
    /// denominator) around the |1> state.
    void R1Dyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    /// Bitwise x axis rotation gate - Rotates each bit from "start" for "length" as e^(-i*\theta/2) around the Pauli x
    /// axis.
    void RX(double radians, bitLenInt start, bitLenInt length);
    /// Bitwise dyadic x axis rotation gate - Rotates each bit from "start" for "length" as e^(i*(M_PI * numerator) /
    /// denominator) around the Pauli x axis.
    void RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    /// Bitwise controlled x axis rotation gate - For each bit pair in "length," if "control" bit is set to 1, rotates
    /// "target" as e^(-i*\theta/2) around the Pauli x axis.
    void CRX(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise dyadic controlled x axis rotation gate - For each bit pair in "length," if "control" bit is set to 1,
    /// rotates "target" as e^(i*(M_PI * numerator) / denominator) around the Pauli x axis.
    void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise y axis rotation gate - Rotates each bit from "start" for "length" as e^(-i*\theta/2) around the Pauli y
    /// axis.
    void RY(double radians, bitLenInt start, bitLenInt length);
    /// Bitwise dyadic y axis rotation gate - Rotates each bit from "start" for "length" as e^(i*(M_PI * numerator) /
    /// denominator) around the Pauli y axis.
    void RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    /// Bitwise controlled y axis rotation gate - For each bit pair in "length," if "control" bit is set to 1, rotates
    /// "target" as e^(-i*\theta/2) around the Pauli y axis.
    void CRY(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise dyadic controlled y axis rotation gate - For each bit pair in "length," if "control" bit is set to 1,
    /// rotates "target" as e^(i*(M_PI * numerator) / denominator) around the Pauli y axis.
    void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise z axis rotation gate - Rotates each bit from "start" for "length" as e^(-i*\theta/2) around the Pauli z
    /// axis.
    void RZ(double radians, bitLenInt start, bitLenInt length);
    /// Bitwise dyadic z axis rotation gate - Rotates each bit from "start" for "length" as e^(i*(M_PI * numerator) /
    /// denominator) around the Pauli z axis.
    void RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    /// Bitwise controlled z axis rotation gate - For each bit pair in "length," if "control" bit is set to 1, rotates
    /// "target" as e^(-i*\theta/2) around the Pauli z axis.
    void CRZ(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise dyadic controlled z axis rotation gate - For each bit pair in "length," if "control" bit is set to 1,
    /// rotates "target" as e^(i*(M_PI * numerator) / denominator) around the Pauli z axis.
    void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise controlled |1> axis rotation gate - For each bit pair in "length," if "control" bit is set to 1, rotates
    /// "target" as e^(-i*\theta/2) around the |1> state.
    void CRT(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise dyadic controlled |1> axis rotation gate - For each bit pair in "length," if "control" bit is set to 1,
    /// rotates "target" as e^(i*(M_PI * numerator) / denominator) around the |1> state.
    void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise controlled y for registers of "length" bits
    void CY(bitLenInt control, bitLenInt target, bitLenInt length);
    /// Bitwise controlled z for registers of "length" bits
    void CZ(bitLenInt control, bitLenInt target, bitLenInt length);

    /// Arithmetic shift left, with last 2 bits as sign and carry
    void ASL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Arithmetic shift right, with last 2 bits as sign and carry
    void ASR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Logical shift left, filling the extra bits with |0>
    void LSL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Logical shift right, filling the extra bits with |0>
    void LSR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// "Circular shift left" - shift bits left, and carry last bits.
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// "Circular shift right" - shift bits right, and carry first bits.
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Add integer (without sign)
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Add integer (without sign, with carry)
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Add a classical integer to the register, with sign and without carry.
    /**
     * Add a classical integer to the register, with sign and without carry. Because the register length is an arbitrary
     * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified
     * as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
     */
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /// Add a classical integer to the register, with sign and with carry.
    /**
     * Add a classical integer to the register, with sign and with carry. If oveflow is set, flip phase on overflow.
     * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is
     * variable. Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be
     * set at the appropriate position before the cast.
     */
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    /// Add a classical integer to the register, with sign and with (phase-based) carry.
    /**
     * Add a classical integer to the register, with sign and with carry. Always flip phase on overflow.
     * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is
     * variable. Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be
     * set at the appropriate position before the cast.
     */
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Add classical BCD integer (without sign)
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Add classical BCD integer (without sign, with carry)
    void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Subtract classical integer (without sign)
    void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);

    /// Subtract classical integer (without sign, with carry)
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Subtract a classical integer from the register, with sign and without carry.
    /**
     * Subtract a classical integer from the register, with sign and without carry. Because the register length is an
     * arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
     * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before
     * the cast.
     */
    void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /// Subtract a classical integer from the register, with sign and with carry.
    /**
     * Subtract a classical integer from the register, with sign and with carry. If oveflow is set, flip phase on
     * overflow.Because the register length is an arbitrary number of bits, the sign bit position on the integer to
     * add is variable. Hence, the integer to add is specified as cast to an unsigned format, with the sign bit
     * assumed to be set at the appropriate position before the cast.
     */
    void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);

    /// Subtract a classical integer from the register, with sign and with carry.
    /**
     * Subtract a classical integer from the register, with sign and with carry. Flip phase on overflow. Because the
     * register length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence,
     * the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the
     * appropriate position before the cast.
     */
    void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Subtract BCD integer (without sign)
    void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Subtract BCD integer (without sign, with carry)
    void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /*
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    // virtual void ADD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /*
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit.
     */
    // void ADDC(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt
    // carryIndex);

    /*
     * Add signed integer of "length" bits in "inStart" to signed integer of "length" bits in "inOutStart," and store
     * result in "inOutStart." Set overflow bit when input to output wraps past minimum or maximum integer.
     */
    // void ADDS(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt overflowIndex);

    /*
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit. Set overflow for
     * signed addition if result wraps past the minimum or maximum signed integer.
     */
    // void ADDSC(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length,
    //    const bitLenInt overflowIndex, const bitLenInt carryIndex);

    /*
     * Add BCD number of "length" bits in "inStart" to BCD number of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    // void ADDBCD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /*
     * Add BCD number of "length" bits in "inStart" to BCD number of "length" bits in "inOutStart," and store result in
     * "inOutStart," with carry in/out.
     */
    // void ADDBCDC(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt carryIndex);

    /*
     * Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    // virtual void SUB(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length);

    /*
     * Subtract BCD number of "length" bits in "inStart" from BCD number of "length" bits in "inOutStart," and store
     * result in "inOutStart."
     */
    // void SUBBCD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /*
     * Subtract BCD number of "length" bits in "inStart" from BCD number of "length" bits in "inOutStart," and store
     * result in "inOutStart," with carry in/out.
     */
    // void SUBBCDC(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt carryIndex);

    /*
     * Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit.
     */
    // void SUBC(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length, const bitLenInt carryIndex);

    /*
     * Subtract signed integer of "length" bits in "inStart" from signed integer of "length" bits in "inOutStart," and
     * store result in "inOutStart." Set overflow bit when input to output wraps past minimum or maximum integer.
     */
    // void SUBS(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt overflowIndex);

    /*
     *Subtract integer of "length" bits in "inStart" from integer of "length" bits in "inOutStart," and store result
     * in "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit. Set overflow for
     * signed addition if result wraps past the minimum or maximum signed integer.
     */
    // void SUBSC(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length, const bitLenInt
    // overflowIndex,
    //    const bitLenInt carryIndex);

    /// Quantum Fourier Transform - Apply the quantum Fourier transform to the register.
    void QFT(bitLenInt start, bitLenInt length);

    /// "Entangled Hadamard" - perform an operation on two entangled registers like a bitwise Hadamard on a single
    /// unentangled register.
    void EntangledH(bitLenInt targetStart, bitLenInt entangledStart, bitLenInt length);

    /// For chips with a zero flag, apply a Z to the zero flag, entangled with the state where the register equals zero.
    void SetZeroFlag(bitLenInt start, bitLenInt length, bitLenInt zeroFlag);

    /// For chips with a zero flag, flip the phase of the state where the register equals zero.
    void SetZeroFlag(bitLenInt start, bitLenInt length);

    /// For chips with a sign flag, apply a Z to the sign flag, entangled with the states where the register is
    /// negative.
    void SetSignFlag(bitLenInt toTest, bitLenInt toSet);

    /// For chips with a sign flag, flip the phase of states where the register is negative.
    void SetSignFlag(bitLenInt toTest);

    /// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
    void SetLessThanFlag(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);

    /// Phase flip always - equivalent to Z X Z X on any bit in the CoherentUnit
    void PhaseFlip();

    /// Set register bits to given permutation
    void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);

    /// Measure permutation state of a register
    bitCapInt MReg(bitLenInt start, bitLenInt length);

    /// Measure permutation state of an 8 bit register
    unsigned char MReg8(bitLenInt start);

    /// Set 8 bit register bits based on read from classical memory
    unsigned char SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values);

    /// Add based on an indexed load from classical memory
    unsigned char AdcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

    /// Subtract based on an indexed load from classical memory
    unsigned char SbcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

protected:
    double runningNorm;
    bitLenInt qubitCount;
    bitCapInt maxQPower;
    std::unique_ptr<Complex16[]> stateVec;

    std::default_random_engine rand_generator;
    std::uniform_real_distribution<double> rand_distribution;

    virtual void ResetStateVec(std::unique_ptr<Complex16[]> nStateVec);
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm);
    void ApplySingleBit(bitLenInt qubitIndex, const Complex16* mtrx, bool doCalcNorm);
    void ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm);
    void ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm);
    void Carry(bitLenInt integerStart, bitLenInt integerLength, bitLenInt carryBit);
    void NormalizeState();
    void Reverse(bitLenInt first, bitLenInt last);
    void UpdateRunningNorm();
};

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride);

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride);

} // namespace Qrack
