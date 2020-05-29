//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.
#pragma once

#define _USE_MATH_DEFINES

#include <ctime>
#include <map>
#include <math.h>
#include <memory>
#include <vector>

#if ENABLE_UINT128
#include <ostream>
#endif

#include "common/parallel_for.hpp"
#include "common/qrack_types.hpp"
#include "common/rdrandwrapper.hpp"
#include "hamiltonian.hpp"

namespace Qrack {

// These are utility functions defined in qinterface/protected.cpp:
unsigned char* cl_alloc(size_t ucharCount);
void cl_free(void* toFree);
void mul2x2(complex* left, complex* right, complex* out);
void exp2x2(complex* matrix2x2, complex* outMatrix2x2);
void log2x2(complex* matrix2x2, complex* outMatrix2x2);
bool isOverflowAdd(bitCapInt inOutInt, bitCapInt inInt, const bitCapInt& signMask, const bitCapInt& lengthPower);
bool isOverflowSub(bitCapInt inOutInt, bitCapInt inInt, const bitCapInt& signMask, const bitCapInt& lengthPower);
bitCapInt pushApartBits(const bitCapInt& perm, const bitCapInt* skipPowers, const bitLenInt skipPowersCount);
bitCapInt intPow(bitCapInt base, bitCapInt power);
bitCapIntOcl intPowOcl(bitCapIntOcl base, bitCapIntOcl power);
#if ENABLE_UINT128
std::ostream& operator<<(std::ostream& left, __uint128_t right);
#endif
inline bitCapInt pow2(const bitLenInt& p) { return ONE_BCI << p; }
inline bitCapIntOcl pow2Ocl(const bitLenInt& p) { return ONE_BCI << p; }
inline bitCapInt pow2Mask(const bitLenInt& p) { return (ONE_BCI << p) - ONE_BCI; }
inline bitCapIntOcl pow2MaskOcl(const bitLenInt& p) { return (ONE_BCI << p) - ONE_BCI; }
inline bitCapInt bitSlice(const bitLenInt& bit, const bitCapInt& source) { return (ONE_BCI << bit) & source; }
inline bitCapIntOcl bitSliceOcl(const bitLenInt& bit, const bitCapIntOcl& source) { return (ONE_BCI << bit) & source; }
inline bitCapInt bitRegMask(const bitLenInt& start, const bitLenInt& length)
{
    return ((ONE_BCI << length) - ONE_BCI) << start;
}
inline bitCapIntOcl bitRegMaskOcl(const bitLenInt& start, const bitLenInt& length)
{
    return ((ONE_BCI << length) - ONE_BCI) << start;
}
// Source: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const bitCapInt& x) { return ((x != 0U) && !(x & (x - ONE_BCI))); }

class QInterface;
typedef std::shared_ptr<QInterface> QInterfacePtr;

/**
 * Enumerated list of supported engines.
 *
 * Use QINTERFACE_OPTIMAL for the best supported engine.
 */
enum QInterfaceEngine {

    /**
     * Create a QEngineCPU leveraging only local CPU and memory resources.
     */
    QINTERFACE_CPU = 0,

    /**
     * Create a QEngineOCL, leveraging OpenCL hardware to increase the speed of certain calculations.
     */
    QINTERFACE_OPENCL,

    /**
     * Create a QUnit, which utilizes other QInterface classes to minimize the amount of work that's needed for any
     * given operation based on the entanglement of the bits involved.
     *
     * This, combined with QINTERFACE_OPTIMAL, is the recommended object to use as a library
     * consumer.
     */
    QINTERFACE_QUNIT,

    /**
     * Create a QUnitMulti, which distributes the explicitly separated "shards" of a QUnit across available OpenCL
     * devices.
     */
    QINTERFACE_QUNIT_MULTI,

    QINTERFACE_FIRST = QINTERFACE_CPU,
#if ENABLE_OPENCL
    QINTERFACE_OPTIMAL = QINTERFACE_OPENCL,
#else
    QINTERFACE_OPTIMAL = QINTERFACE_CPU,
#endif

    QINTERFACE_MAX
};

/**
 * A "Qrack::QInterface" is an abstract interface exposing qubit permutation
 * state vector with methods to operate on it as by gates and register-like
 * instructions.
 *
 * See README.md for an overview of the algorithms Qrack employs.
 */
class QInterface {
protected:
    bitLenInt qubitCount;
    bitCapInt maxQPower;
    uint32_t randomSeed;
    qrack_rand_gen_ptr rand_generator;
    std::uniform_real_distribution<real1> rand_distribution;
    std::shared_ptr<RdRandom> hardware_rand_generator;
    bool doNormalize;
    bool randGlobalPhase;
    real1 amplitudeFloor;

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = pow2(qubitCount);
    }

    static inline bitLenInt log2(bitCapInt n)
    {
        bitLenInt pow = 0;
        bitCapInt p = n >> 1U;
        while (p != 0) {
            p >>= 1U;
            pow++;
        }
        return pow;
    }

    // Compilers have difficulty figuring out types and overloading if the "norm" handle is passed to std::transform. If
    // you need a safe pointer to norm(), try this:
    static inline real1 normHelper(complex c) { return norm(c); }

    static inline real1 clampProb(real1 toClamp)
    {
        if (toClamp < ZERO_R1) {
            toClamp = ZERO_R1;
        }
        if (toClamp > ONE_R1) {
            toClamp = ONE_R1;
        }
        return toClamp;
    }

    template <typename GateFunc> void ControlledLoopFixture(bitLenInt length, GateFunc gate);

    void FreeAligned(void* toFree)
    {
        if (toFree) {
#if defined(_WIN32)
            _aligned_free(toFree);
#else
            free(toFree);
#endif
        }
        toFree = NULL;
    }

    bool IsIdentity(const complex* mtrx, const bool isControlled = false);

public:
    QInterface(bitLenInt n, qrack_rand_gen_ptr rgp = nullptr, bool doNorm = false, bool useHardwareRNG = true,
        bool randomGlobalPhase = true, real1 norm_thresh = REAL1_DEFAULT_ARG)
        : rand_distribution(0.0, 1.0)
        , hardware_rand_generator(NULL)
        , doNormalize(doNorm)
        , randGlobalPhase(randomGlobalPhase)
        , amplitudeFloor(norm_thresh)
    {
        SetQubitCount(n);

        if (useHardwareRNG) {
            hardware_rand_generator = std::make_shared<RdRandom>();
            if (!(hardware_rand_generator->SupportsRDRAND())) {
                hardware_rand_generator = NULL;
            }
        }

        if (rgp == NULL) {
            rand_generator = std::make_shared<qrack_rand_gen>();
            randomSeed = std::time(0);
            SetRandomSeed(randomSeed);
        } else {
            rand_generator = rgp;
        }

        if (amplitudeFloor < 0) {
            amplitudeFloor = min_norm;
        }
    }

    QInterface()
    {
        // Intentionally left blank
    }

    /** Destructor of QInterface */
    virtual ~QInterface(){};

    virtual void SetRandomSeed(uint32_t seed)
    {
        if (rand_generator != NULL) {
            rand_generator->seed(seed);
        }
    }

    /** Get the count of bits in this register */
    bitLenInt GetQubitCount() { return qubitCount; }

    /** Get the maximum number of basis states, namely \f$ n^2 \f$ for \f$ n \f$ qubits*/
    bitCapInt GetMaxQPower() { return maxQPower; }

    /** Generate a random real number between 0 and 1 */
    real1 Rand()
    {
        if (hardware_rand_generator != NULL) {
            return hardware_rand_generator->Next();
        } else {
            return rand_distribution(*rand_generator);
        }
    }

    /** Set an arbitrary pure quantum state representation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void SetQuantumState(const complex* inputState) = 0;

    /** Get the pure quantum state representation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void GetQuantumState(complex* outputState) = 0;

    /** Get the pure quantum state representation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void GetProbs(real1* outputProbs) = 0;

    /** Get the representational amplitude of a full permutation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual complex GetAmplitude(bitCapInt perm) = 0;

    /** Sets the representational amplitude of a full permutation
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void SetAmplitude(bitCapInt perm, complex amp) = 0;

    /** Set to a specific permutation */
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG) = 0;

    /**
     * Combine another QInterface with this one, after the last bit index of
     * this one.
     *
     * "Compose" combines the quantum description of state of two independent
     * QInterface objects into one object, containing the full permutation
     * basis of the full object. The "inputState" bits are added after the last
     * qubit index of the QInterface to which we "Compose." Informally,
     * "Compose" is equivalent to "just setting another group of qubits down
     * next to the first" without interacting them. Schroedinger's equation can
     * form a description of state for two independent subsystems at once or
     * "separable quantum subsystems" without interacting them. Once the
     * description of state of the independent systems is combined, we can
     * interact them, and we can describe their entanglements to each other, in
     * which case they are no longer independent. A full entangled description
     * of quantum state is not possible for two independent quantum subsystems
     * until we "Compose" them.
     *
     * "Compose" multiplies the probabilities of the indepedent permutation
     * states of the two subsystems to find the probabilites of the entire set
     * of combined permutations, by simple combinatorial reasoning. If the
     * probablity of the "left-hand" subsystem being in |00> is 1/4, and the
     * probablity of the "right-hand" subsystem being in |101> is 1/8, than the
     * probability of the combined |00101> permutation state is 1/32, and so on
     * for all permutations of the new combined state.
     *
     * If the programmer doesn't want to "cheat" quantum mechanically, then the
     * original copy of the state which is duplicated into the larger
     * QInterface should be "thrown away" to satisfy "no clone theorem." This
     * is not semantically enforced in Qrack, because optimization of an
     * emulator might be acheived by "cloning" "under-the-hood" while only
     * exposing a quantum mechanically consistent API or instruction set.
     *
     * Returns the quantum bit offset that the QInterface was appended at, such
     * that bit 5 in toCopy is equal to offset+5 in this object.
     */
    virtual bitLenInt Compose(QInterfacePtr toCopy) = 0;
    virtual std::map<QInterfacePtr, bitLenInt> Compose(std::vector<QInterfacePtr> toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start) = 0;

    /**
     * Minimally decompose a set of contiguous bits from the separably composed unit,
     * into "destination"
     *
     * Minimally decompose a set of contigious bits from the separably composed unit.
     * The length of this separable unit is reduced by the length of bits decomposed, and the bits removed are output in
     * the destination QInterface pointer. The destination object must be initialized to the correct number of bits, in
     * 0 permutation state. For quantum mechanical accuracy, the bit set removed and the bit set left behind should be
     * quantum mechanically "separable."
     *
     * Like how "Compose" is like "just setting another group of qubits down
     * next to the first," <b><i>if two sets of qubits are not
     * entangled,</i></b> then "Decompose" is like "just moving a few qubits
     * away from the rest." Schroedinger's equation does not require bits to be
     * explicitly interacted in order to describe their permutation basis, and
     * the descriptions of state of <b>separable</b> subsystems, those which
     * are not entangled with other subsystems, are just as easily removed from
     * the description of state. (This is equivalent to a "Schmidt decomposition.")
     *
     * If we have for example 5 qubits, and we wish to separate into "left" and
     * "right" subsystems of 3 and 2 qubits, we sum probabilities of one
     * permutation of the "left" three over ALL permutations of the "right"
     * two, for all permutations, and vice versa, like so:
     *
     * \f$
     *     P(|1000>|xy>) = P(|1000 00>) + P(|1000 10>) + P(|1000 01>) + P(|1000 11>).
     * \f$
     *
     * If the subsystems are not "separable," i.e. if they are entangled, this
     * operation is not well-motivated, and its output is not necessarily
     * defined. (The summing of probabilities over permutations of subsytems
     * will be performed as described above, but this is not quantum
     * mechanically meaningful.) To ensure that the subsystem is "separable,"
     * i.e. that it has no entanglements to other subsystems in the
     * QInterface, it can be measured with M(), or else all qubits <i>other
     * than</i> the subsystem can be measured.
     */
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest) = 0;

    /**
     * Minimally decompose a set of contiguous bits from the separably composed unit,
     * and discard the separable bits from index "start" for "length."
     *
     * Minimally decompose a set of contigious bits from the separably composed unit.
     * The length of this separable unit is reduced by the length of bits decomposed, and the bits removed are output in
     * the destination QInterface pointer. The destination object must be initialized to the correct number of bits, in
     * 0 permutation state. For quantum mechanical accuracy, the bit set removed and the bit set left behind should be
     * quantum mechanically "separable."
     *
     * Like how "Compose" is like "just setting another group of qubits down
     * next to the first," <b><i>if two sets of qubits are not
     * entangled,</i></b> then "Decompose" is like "just moving a few qubits
     * away from the rest." Schroedinger's equation does not require bits to be
     * explicitly interacted in order to describe their permutation basis, and
     * the descriptions of state of <b>separable</b> subsystems, those which
     * are not entangled with other subsystems, are just as easily removed from
     * the description of state. (This is equivalent to a "Schmidt decomposition.")
     *
     * If we have for example 5 qubits, and we wish to separate into "left" and
     * "right" subsystems of 3 and 2 qubits, we sum probabilities of one
     * permutation of the "left" three over ALL permutations of the "right"
     * two, for all permutations, and vice versa, like so:
     *
     * \f$
     *     P(|1000>|xy>) = P(|1000 00>) + P(|1000 10>) + P(|1000 01>) + P(|1000 11>).
     * \f$
     *
     * If the subsystems are not "separable," i.e. if they are entangled, this
     * operation is not well-motivated, and its output is not necessarily
     * defined. (The summing of probabilities over permutations of subsytems
     * will be performed as described above, but this is not quantum
     * mechanically meaningful.) To ensure that the subsystem is "separable,"
     * i.e. that it has no entanglements to other subsystems in the
     * QInterface, it can be measured with M(), or else all qubits <i>other
     * than</i> the subsystem can be measured.
     */
    virtual void Dispose(bitLenInt start, bitLenInt length) = 0;

    /**
     * Dispose a a contiguous set of qubits that are already in a permutation eigenstate.
     */
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm) = 0;

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    /**
     * Apply an arbitrary single bit unitary transformation.
     */
    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubitIndex) = 0;

    /**
     * Apply an arbitrary single bit unitary transformation, with arbitrary control bits.
     */
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx) = 0;

    /**
     * Apply an arbitrary single bit unitary transformation, with arbitrary (anti-)control bits.
     */
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx) = 0;

    /**
     * Apply a single bit transformation that only effects phase.
     */
    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex);

    /**
     * Apply a single bit transformation that reverses bit probability and might effect phase.
     */
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex);

    /**
     * Apply a single bit transformation that only effects phase, with arbitrary control bits.
     */
    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);

    /**
     * Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary control
     * bits.
     */
    virtual void ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);

    /**
     * Apply a single bit transformation that only effects phase, with arbitrary (anti-)control bits.
     */
    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);

    /**
     * Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary
     * (anti-)control bits.
     */
    virtual void ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);
    /**
     * Apply a "uniformly controlled" arbitrary single bit unitary transformation. (See
     * https://arxiv.org/abs/quant-ph/0312218)
     *
     * A different unitary 2x2 complex matrix is associated with each permutation of the control bits. The first control
     * bit index in the "controls" array is the least significant bit of the permutation, proceeding to the most
     * significant bit. "mtrxs" is a flat (1-dimensional) array where each subsequent set of 4 components is an
     * arbitrary 2x2 single bit gate associated with the next permutation of the control bits, starting from 0. All
     * combinations of control bits apply one of the 4 component (flat 2x2) matrices. For k control bits, there are
     * therefore 4 * 2^k complex components in "mtrxs," representing 2^k complex matrices of 2x2 components. (The
     * component ordering in each matrix is the same as all other gates with an arbitrary 2x2 applied to a single bit,
     * such as Qrack::ApplySingleBit.)
     */

    virtual void UniformlyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const complex* mtrxs)
    {
        UniformlyControlledSingleBit(controls, controlLen, qubitIndex, mtrxs, NULL, 0, 0);
    }
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);

    /**
     * To define a Hamiltonian, give a vector of controlled single bit gates ("HamiltonianOp" instances) that are
     * applied by left-multiplication in low-to-high vector index order on the state vector.
     *
     * \warning Hamiltonian components might not commute.
     *
     * As a general point of linear algebra, where A and B are linear operators, \f{equation}{e^{i (A + B) t} = e^{i A
     * t} e^{i B t} \f} might NOT hold, if the operators A and B do not commute. As a rule of thumb, A will commute
     * with B at least in the case that A and B act on entirely different sets of qubits. However, for defining the
     * intended Hamiltonian, the programmer can be guaranteed that the exponential factors will be applied
     * right-to-left, by left multiplication, in the order \f{equation}{ e^{-i H_{N - 1} t} e^{-i H_{N - 2} t} \ldots
     * e^{-i H_0 t} \left|\psi \rangle\right. .\f} (For example, if A and B are single bit gates acting on the same
     * bit, form their composition into one gate by the intended right-to-left fusion and apply them as a single
     * HamiltonianOp.)
     */
    virtual void TimeEvolve(Hamiltonian h, real1 timeDiff);

    /**
     * Apply a swap with arbitrary control bits.
     */
    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    /**
     * Apply a swap with arbitrary (anti) control bits.
     */
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    /**
     * Apply a square root of swap with arbitrary control bits.
     */
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    /**
     * Apply a square root of swap with arbitrary (anti) control bits.
     */
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    /**
     * Apply an inverse square root of swap with arbitrary control bits.
     */
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    /**
     * Apply an inverse square root of swap with arbitrary (anti) control bits.
     */
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    /**
     * Doubly-controlled NOT gate
     *
     * If both controls are set to 1, the target bit is NOT-ed or X-ed.
     */
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);

    /**
     * Anti doubly-controlled NOT gate
     *
     * If both controls are set to 0, the target bit is NOT-ed or X-ed.
     */
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);

    /**
     * Controlled NOT gate
     *
     * If the control is set to 1, the target bit is NOT-ed or X-ed.
     */
    virtual void CNOT(bitLenInt control, bitLenInt target);

    /**
     * Anti controlled NOT gate
     *
     * If the control is set to 0, the target bit is NOT-ed or X-ed.
     */
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);

    /**
     * General unitary gate
     *
     * Applies a gate guaranteed to be unitary, from three angles, as commonly defined, spanning all possible single bit
     * unitary gates, (up to a global phase factor which has no effect on Hermitian operator expectation values).
     */
    virtual void U(bitLenInt target, real1 theta, real1 phi, real1 lambda);

    /**
     * 2-parameter unitary gate
     *
     * Applies a gate guaranteed to be unitary, from two angles, as commonly defined.
     */
    virtual void U2(bitLenInt target, real1 phi, real1 lambda) { U(target, M_PI / 2, phi, lambda); }

    /**
     * Hadamard gate
     *
     * Applies a Hadamard gate on qubit at "qubitIndex."
     */
    virtual void H(bitLenInt qubitIndex);

    /**
     * Square root of Hadamard gate
     *
     * Applies the square root of the Hadamard gate on qubit at "qubitIndex."
     */
    virtual void SqrtH(bitLenInt qubitIndex);

    /**
     * Measurement gate
     *
     * Measures the qubit at "qubitIndex" and returns either "true" or "false."
     * (This "gate" breaks unitarity.)
     *
     * All physical evolution of a quantum state should be "unitary," except
     * measurement. Measurement of a qubit "collapses" the quantum state into
     * either only permutation states consistent with a |0> state for the bit,
     * or else only permutation states consistent with a |1> state for the bit.
     * Measurement also effectively multiplies the overall quantum state vector
     * of the system by a random phase factor, equiprobable over all possible
     * phase angles.
     *
     * Effectively, when a bit measurement is emulated, Qrack calculates the
     * norm of all permutation state components, to find their respective
     * probabilities. The probabilities of all states in which the measured
     * bit is "0" can be summed to give the probability of the bit being "0,"
     * and separately the probabilities of all states in which the measured
     * bit is "1" can be summed to give the probability of the bit being "1."
     * To simulate measurement, a random float between 0 and 1 is compared to
     * the sum of the probability of all permutation states in which the bit
     * is equal to "1". Depending on whether the random float is higher or
     * lower than the probability, the qubit is determined to be either |0> or
     * |1>, (up to phase). If the bit is determined to be |1>, then all
     * permutation eigenstates in which the bit would be equal to |0> have
     * their probability set to zero, and vice versa if the bit is determined
     * to be |0>. Then, all remaining permutation states with nonzero
     * probability are linearly rescaled so that the total probability of all
     * permutation states is again "normalized" to exactly 100% or 1, (within
     * double precision rounding error). Physically, the act of measurement
     * should introduce an overall random phase factor on the state vector,
     * which is emulated by generating another constantly distributed random
     * float to select a phase angle between 0 and 2 * Pi.
     *
     * Measurement breaks unitary evolution of state. All quantum gates except
     * measurement should generally act as a unitary matrix on a permutation
     * state vector. (Note that Boolean comparison convenience methods in Qrack
     * such as "AND," "OR," and "XOR" employ the measurement operation in the
     * act of first clearing output bits before filling them with the result of
     * comparison, and these convenience methods therefore break unitary
     * evolution of state, but in a physically realistic way. Comparable
     * unitary operations would be performed with a combination of X and CCNOT
     * gates, also called "Toffoli" gates, but the output bits would have to be
     * assumed to be in a known fixed state, like all |0>, ahead of time to
     * produce unitary logical comparison operations.)
     */
    virtual bool M(bitLenInt qubitIndex) { return ForceM(qubitIndex, false, false); };

    /**
     * Act as if is a measurement was applied, except force the (usually random) result
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true) = 0;

    /**
     * S gate
     *
     * Applies a 1/4 phase rotation to the qubit at "qubitIndex."
     */
    virtual void S(bitLenInt qubitIndex);

    /**
     * Inverse S gate
     *
     * Applies an inverse 1/4 phase rotation to the qubit at "qubitIndex."
     */
    virtual void IS(bitLenInt qubitIndex);

    /**
     * T gate
     *
     * Applies a 1/8 phase rotation to the qubit at "qubitIndex."
     */
    virtual void T(bitLenInt qubitIndex);

    /**
     * Inverse T gate
     *
     * Applies an inverse 1/8 phase rotation to the qubit at "qubitIndex."
     */
    virtual void IT(bitLenInt qubitIndex);

    /**
     * "PhaseRootN" gate
     *
     * Applies a 1/(2^N) phase rotation to the qubit at "qubitIndex."
     */
    virtual void PhaseRootN(bitLenInt n, bitLenInt qubitIndex);

    /**
     * Inverse "PhaseRootN" gate
     *
     * Applies an inverse 1/(2^N) phase rotation to the qubit at "qubitIndex."
     */
    virtual void IPhaseRootN(bitLenInt n, bitLenInt qubitIndex);

    /**
     * X gate
     *
     * Applies the Pauli "X" operator to the qubit at "qubitIndex." The Pauli
     * "X" operator is equivalent to a logical "NOT."
     */
    virtual void X(bitLenInt qubitIndex);

    /**
     * Y gate
     *
     * Applies the Pauli "Y" operator to the qubit at "qubitIndex." The Pauli
     * "Y" operator is similar to a logical "NOT" with permutation phase
     * effects.
     */
    virtual void Y(bitLenInt qubitIndex);

    /**
     * Z gate
     *
     * Applies the Pauli "Z" operator to the qubit at "qubitIndex." The Pauli
     * "Z" operator reverses the phase of |1> and leaves |0> unchanged.
     */
    virtual void Z(bitLenInt qubitIndex);

    /**
     * Square root of X gate
     *
     * Applies the square root of the Pauli "X" operator to the qubit at "qubitIndex." The Pauli
     * "X" operator is equivalent to a logical "NOT."
     */
    virtual void SqrtX(bitLenInt qubitIndex);

    /**
     * Inverse square root of X gate
     *
     * Applies the (by convention) inverse square root of the Pauli "X" operator to the qubit at "qubitIndex." The Pauli
     * "X" operator is equivalent to a logical "NOT."
     */
    virtual void ISqrtX(bitLenInt qubitIndex);

    /**
     * Phased square root of X gate
     *
     * Applies T.SqrtX.IT to the qubit at "qubitIndex."
     */
    virtual void SqrtXConjT(bitLenInt qubitIndex);

    /**
     * Inverse phased square root of X gate
     *
     * Applies IT.ISqrtX.T to the qubit at "qubitIndex."
     */
    virtual void ISqrtXConjT(bitLenInt qubitIndex);

    /**
     * Square root of Y gate
     *
     * Applies the square root of the Pauli "Y" operator to the qubit at "qubitIndex." The Pauli
     * "Y" operator is similar to a logical "NOT" with permutation phase
     * effects.
     */
    virtual void SqrtY(bitLenInt qubitIndex);

    /**
     * Square root of Y gate
     *
     * Applies the (by convention) inverse square root of the Pauli "Y" operator to the qubit at "qubitIndex." The Pauli
     * "Y" operator is similar to a logical "NOT" with permutation phase
     * effects.
     */
    virtual void ISqrtY(bitLenInt qubitIndex);

    /**
     * Controlled Y gate
     *
     * If the "control" bit is set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    virtual void CY(bitLenInt control, bitLenInt target);

    /**
     * Controlled Z gate
     *
     * If the "control" bit is set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CZ(bitLenInt control, bitLenInt target);

    /**
     * Doubly-Controlled Z gate
     *
     * If both "control" bits are set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target);

    /**
     * Controlled H gate
     *
     * If the "control" bit is set to 1, then the "H" Walsh-Hadamard transform operator is applied
     * to "target."
     */
    virtual void CH(bitLenInt control, bitLenInt target);

    /**
     * Controlled S gate
     *
     * If the "control" bit is set to 1, then the S gate is applied
     * to "target."
     */
    virtual void CS(bitLenInt control, bitLenInt target);

    /**
     * Controlled inverse S gate
     *
     * If the "control" bit is set to 1, then the inverse S gate is applied
     * to "target."
     */
    virtual void CIS(bitLenInt control, bitLenInt target);

    /**
     * Controlled T gate
     *
     * If the "control" bit is set to 1, then the T gate is applied
     * to "target."
     */
    virtual void CT(bitLenInt control, bitLenInt target);

    /**
     * Controlled inverse T gate
     *
     * If the "control" bit is set to 1, then the inverse T gate is applied
     * to "target."
     */
    virtual void CIT(bitLenInt control, bitLenInt target);

    /**
     * Controlled "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target);

    /**
     * Controlled inverse "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the inverse "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target);

    /** @} */

    /**
     * \defgroup LogicGates Logic Gates
     *
     * Each bit is paired with a CL* variant that utilizes a classical bit as
     * an input.
     *
     * @{
     */

    /**
     * Quantum analog of classical "AND" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "OR" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XOR" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);

    /**
     *  Quantum analog of classical "AND" gate. Takes one qubit input and one
     *  classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "OR" gate. Takes one qubit input and one
     * classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /**
     * Quantum analog of classical "XOR" gate. Takes one qubit input and one
     * classical bit input. (Assumes the outputBit is in the 0 state)
     */
    virtual void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /** @} */

    /**
     * \defgroup RotGates Rotational gates:
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     *
     * @{
     */

    /**
     * Phase shift gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around |1> state
     */
    virtual void RT(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction phase shift gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around |1>
     * state.
     */
    virtual void RTDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * X axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli X axis
     */
    virtual void RX(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction X axis rotation gate
     *
     * Rotates \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ on Pauli x axis.
     */
    virtual void RXDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * (Identity) Exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*I} \f$, exponentiation of the identity operator
     */
    virtual void Exp(real1 radians, bitLenInt qubitIndex);

    /**
     *  Imaginary exponentiation of arbitrary 2x2 gate
     *
     * Applies \f$ e^{-i*Op} \f$, where "Op" is a 2x2 matrix, (with controls on the application of the gate).
     */
    virtual void Exp(
        bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, complex* matrix2x2, bool antiCtrled = false);

    /**
     * Dyadic fraction (identity) exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * I / 2^{denomPower}\right) \f$, exponentiation of the identity
     * operator
     */
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpX(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Pauli X exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_x / 2^{denomPower}\right) \f$, exponentiation of the Pauli X
     * operator
     */
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpY(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Pauli Y exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_y / 2^{denomPower}\right) \f$, exponentiation of the Pauli Y
     * operator
     */
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZ(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Pauli Z exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_z / 2^{denomPower}\right) \f$, exponentiation of the Pauli Z
     * operator
     */
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i*\theta/2} \f$ on Pauli x axis.
     */
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli x axis.
     */
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Y axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli y axis.
     */
    virtual void RY(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y axis.
     */
    virtual void RYDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Y axis.
     */
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Apply a "uniformly controlled" rotation of a bit around the Pauli Y axis. (See
     * https://arxiv.org/abs/quant-ph/0312218)
     *
     * A different rotation angle is associated with each permutation of the control bits. The first control bit index
     * in the "controls" array is the least significant bit of the permutation, proceeding to the most significant bit.
     * "angles" is an array where each subsequent component is rotation angle associated with the next permutation of
     * the control bits, starting from 0. All combinations of control bits apply one of rotation angles. For k control
     * bits, there are therefore 2^k real components in "angles."
     */
    virtual void UniformlyControlledRY(
        const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const real1* angles);

    /**
     * Apply a "uniformly controlled" rotation of a bit around the Pauli Z axis. (See
     * https://arxiv.org/abs/quant-ph/0312218)
     *
     * A different rotation angle is associated with each permutation of the control bits. The first control bit index
     * in the "controls" array is the least significant bit of the permutation, proceeding to the most significant bit.
     * "angles" is an array where each subsequent component is rotation angle associated with the next permutation of
     * the control bits, starting from 0. All combinations of control bits apply one of rotation angles. For k control
     * bits, there are therefore 2^k real components in "angles."
     */
    virtual void UniformlyControlledRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const real1* angles);

    /**
     * Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    virtual void RZ(real1 radians, bitLenInt qubitIndex);

    /**
     * Dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z axis.
     */
    virtual void RZDyad(int numerator, int denomPower, bitLenInt qubitIndex);

    /**
     * Controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Zaxis.
     */
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z
     * axis.
     */
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /**
     * Controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i*\theta/2}
     * \f$ around |1> state.
     */

    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target);

    /**
     * Controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$
     * around |1> state.
     */
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target);

    /** @} */

    /**
     * \defgroup RegGates Register-spanning gates
     *
     * Convienence and optimized functions implementing gates are applied from
     * the bit 'start' for 'length' bits for the register.
     *
     * @{
     */

    /** Bitwise general unitary */
    virtual void U(bitLenInt start, bitLenInt length, real1 theta, real1 phi, real1 lambda);

    /** Bitwise 2-parameter unitary */
    virtual void U2(bitLenInt start, bitLenInt length, real1 phi, real1 lambda);

    /** Bitwise Hadamard */
    virtual void H(bitLenInt start, bitLenInt length);

    /** Bitwise square root of Hadamard */
    virtual void SqrtH(bitLenInt start, bitLenInt length);

    /** Bitwise S operator (1/4 phase rotation) */
    virtual void S(bitLenInt start, bitLenInt length);

    /** Bitwise inverse S operator (1/4 phase rotation) */
    virtual void IS(bitLenInt start, bitLenInt length);

    /** Bitwise T operator (1/8 phase rotation) */
    virtual void T(bitLenInt start, bitLenInt length);

    /** Bitwise inverse T operator (1/8 phase rotation) */
    virtual void IT(bitLenInt start, bitLenInt length);

    /** Bitwise "PhaseRootN" operator (1/(2^N) phase rotation) */
    virtual void PhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length);

    /** Bitwise inverse "PhaseRootN" operator (1/(2^N) phase rotation) */
    virtual void IPhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length);

    /** Bitwise Pauli X (or logical "NOT") operator */
    virtual void X(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli Y operator */
    virtual void Y(bitLenInt start, bitLenInt length);

    /** Bitwise square root of Pauli X operator */
    virtual void SqrtX(bitLenInt start, bitLenInt length);

    /** Bitwise inverse square root of Pauli X operator */
    virtual void ISqrtX(bitLenInt start, bitLenInt length);

    /** Bitwise phased square root of Pauli X operator */
    virtual void SqrtXConjT(bitLenInt start, bitLenInt length);

    /** Bitwise inverse phased square root of Pauli X operator */
    virtual void ISqrtXConjT(bitLenInt start, bitLenInt length);

    /** Bitwise square root of Pauli Y operator */
    virtual void SqrtY(bitLenInt start, bitLenInt length);

    /** Bitwise inverse square root of Pauli Y operator */
    virtual void ISqrtY(bitLenInt start, bitLenInt length);

    /** Bitwise Pauli Z operator */
    virtual void Z(bitLenInt start, bitLenInt length);

    /** Bitwise controlled-not */
    virtual void CNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);

    /** Bitwise "anti-"controlled-not */
    virtual void AntiCNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);

    /** Bitwise doubly controlled-not */
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /** Bitwise doubly "anti-"controlled-not */
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /**
     * Bitwise "AND"
     *
     * "AND" registers at "inputStart1" and "inputStart2," of "length" bits,
     * placing the result in "outputStart".
     */
    virtual void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /**
     * Classical bitwise "AND"
     *
     * "AND" registers at "inputStart1" and the classic bits of "classicalInput," of "length" bits,
     * placing the result in "outputStart".
     */
    virtual void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "OR" */
    virtual void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "OR" */
    virtual void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /** Bitwise "XOR" */
    virtual void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);

    /** Classical bitwise "XOR" */
    virtual void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /**
     * Bitwise phase shift gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around |1> state
     */
    virtual void RT(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction phase shift gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around |1>
     * state.
     */
    virtual void RTDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise X axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli X axis
     */
    virtual void RX(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction X axis rotation gate
     *
     * Rotates \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ on Pauli x axis.
     */
    virtual void RXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i*\theta/2} \f$ on Pauli x axis.
     */
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli x axis.
     */
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Y axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli y axis.
     */
    virtual void RY(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    virtual void RYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Y axis.
     */
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    virtual void RZ(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z axis.
     */
    virtual void RZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Zaxis.
     */
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z
     * axis.
     */
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i*\theta/2}
     * \f$ around |1> state.
     */
    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$
     * around |1> state.
     */
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise (identity) exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*I} \f$, exponentiation of the identity operator
     */
    virtual void Exp(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction (identity) exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * I / 2^{denomPower}\right) \f$, exponentiation of the identity
     * operator
     */
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
     */
    virtual void ExpX(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli X exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_x / 2^{denomPower}\right) \f$, exponentiation of the Pauli X
     * operator
     */
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
     */
    virtual void ExpY(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli Y exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_y / 2^{denomPower}\right) \f$, exponentiation of the Pauli Y
     * operator
     */
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
     */
    virtual void ExpZ(real1 radians, bitLenInt start, bitLenInt length);

    /**
     * Bitwise Dyadic fraction Pauli Z exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_z / 2^{denomPower}\right) \f$, exponentiation of the Pauli Z
     * operator
     */
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length);

    /**
     * Bitwise controlled Y gate
     *
     * If the "control" bit is set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    virtual void CY(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled Z gate
     *
     * If the "control" bit is set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CZ(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise doubly-controlled Z gate
     *
     * If both "control" bits are set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    virtual void CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled H gate
     *
     * If the "control" bit is set to 1, then the "H" Walsh-Hadamard transform operator is applied
     * to "target."
     */
    virtual void CH(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled S gate
     *
     * If the "control" bit is set to 1, then the S gate is applied
     * to "target."
     */
    virtual void CS(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled inverse S gate
     *
     * If the "control" bit is set to 1, then the inverse S gate is applied
     * to "target."
     */
    virtual void CIS(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled T gate
     *
     * If the "control" bit is set to 1, then the T gate is applied
     * to "target."
     */
    virtual void CT(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled inverse T gate
     *
     * If the "control" bit is set to 1, then the inverse T gate is applied
     * to "target."
     */
    virtual void CIT(bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length);

    /**
     * Bitwise controlled inverse "PhaseRootN" gate
     *
     * If the "control" bit is set to 1, then the inverse "PhaseRootN" gate is applied
     * to "target."
     */
    virtual void CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length);

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    /** Arithmetic shift left, with last 2 bits as sign and carry */
    virtual void ASL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Arithmetic shift right, with last 2 bits as sign and carry */
    virtual void ASR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift left, filling the extra bits with |0> */
    virtual void LSL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Logical shift right, filling the extra bits with |0> */
    virtual void LSR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Circular shift left - shift bits left, and carry last bits. */
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Circular shift right - shift bits right, and carry first bits. */
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /** Add integer (without sign) */
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;

    /** Add integer (without sign, with controls) */
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen) = 0;

    /** Add integer (without sign, with carry) */
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Add a classical integer to the register, with sign and without carry. */
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) = 0;

    /** Add a classical integer to the register, with sign and with carry. */
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;

    /** Add a classical integer to the register, with sign and with (phase-based) carry. */
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Add classical BCD integer (without sign) */
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;

    /** Add classical BCD integer (without sign, with carry) */
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Subtract classical integer (without sign) */
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);

    /** Subtract classical integer (without sign, with controls) */
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);

    /** Subtract classical integer (without sign, with carry) */
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Subtract a classical integer from the register, with sign and without carry. */
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;

    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Subtract BCD integer (without sign) */
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /** Subtract BCD integer (without sign, with carry) */
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    /** Multiply by integer */
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length) = 0;

    /** Divide by integer */
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length) = 0;

    /** Multiplication modulo N by integer, (out of place) */
    virtual void MULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;

    /** Inverse: multiplication modulo N by integer, (out of place) */
    virtual void IMULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;

    /** Raise a classical base to a quantum power, modulo N, (out of place) */
    virtual void POWModNOut(
        bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;

    /** Controlled multiplication by integer */
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;

    /** Controlled division by power of integer */
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;

    /** Controlled multiplication modulo N by integer, (out of place) */
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;

    /** Inverse: controlled multiplication modulo N by integer, (out of place) */
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;

    /** Controlled, raise a classical base to a quantum power, modulo N, (out of place) */
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;

    /**
     * Quantum analog of classical "Full Adder" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);

    /**
     * Inverse of FullAdd
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert FullAdd.)
     */
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);

    /**
     * Controlled quantum analog of classical "Full Adder" gate
     *
     * (Assumes the outputBit is in the 0 state)
     */
    virtual void CFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
        bitLenInt carryInSumOut, bitLenInt carryOut);

    /**
     * Inverse of CFullAdd
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert CFullAdd.)
     */
    virtual void CIFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
        bitLenInt carryInSumOut, bitLenInt carryOut);

    /**
     * Add a quantum integer to a quantum integer, with carry
     *
     * (Assumes the output register is in the 0 state)
     */
    virtual void ADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry);

    /**
     * Inverse of ADC
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert ADC.)
     */
    virtual void IADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry);

    /**
     * Add a quantum integer to a quantum integer, with carry and with controls
     *
     * (Assumes the output register is in the 0 state)
     */
    virtual void CADC(bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2, bitLenInt output,
        bitLenInt length, bitLenInt carry);

    /**
     * Inverse of CADC
     *
     * (Can be thought of as "subtraction," but with a register convention that the same inputs invert CADC.)
     */
    virtual void CIADC(bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2, bitLenInt output,
        bitLenInt length, bitLenInt carry);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    /** Quantum Fourier Transform - Apply the quantum Fourier transform to the register.
     *
     * "trySeparate" is an optional hit-or-miss optimization, specifically for QUnit types. Our suggestion is, turn it
     * on for speed and memory effciency if you expect the result of the QFT to be in a permutation basis eigenstate.
     * Otherwise, turning it on will probably take longer.
     */
    virtual void QFT(bitLenInt start, bitLenInt length, bool trySeparate = false);

    /** Inverse Quantum Fourier Transform - Apply the inverse quantum Fourier transform to the register.
     *
     * "trySeparate" is an optional hit-or-miss optimization, specifically for QUnit types. Our suggestion is, turn it
     * on for speed and memory effciency if you expect the result of the QFT to be in a permutation basis eigenstate.
     * Otherwise, turning it on will probably take longer.
     */
    virtual void IQFT(bitLenInt start, bitLenInt length, bool trySeparate = false);

    /** Reverse the phase of the state where the register equals zero. */
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) = 0;

    /** The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation. */
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) = 0;

    /** This is an expedient for an adaptive Grover's search for a function's global minimum. */
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length) = 0;

    /** Phase flip always - equivalent to Z X Z X on any bit in the QInterface */
    virtual void PhaseFlip() = 0;

    /** Set register bits to given permutation */
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);

    /** Measure permutation state of a register */
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length) { return ForceMReg(start, length, 0, false); }

    /**
     * Act as if is a measurement was applied, except force the (usually random) result
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true);

    /** Measure bits with indices in array, and return a mask of the results */
    virtual bitCapInt M(const bitLenInt* bits, const bitLenInt& length) { return ForceM(bits, length, NULL); }

    /** Measure bits with indices in array, and return a mask of the results */
    virtual bitCapInt ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values, bool doApply = true);

    /**
     * Set 8 bit register bits by a superposed index-offset-based read from
     * classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits are first cleared,
     * then the separable |input, 00000000> permutation state is mapped to
     * |input, values[input]>, with "values[input]" placed in the "outputStart"
     * register. FOR BEST EFFICIENCY, the "values" array should be allocated aligned to a 64-byte boundary. (See the
     * unit tests suite code for an example of how to align the allocation.)
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM.
     *
     * The physical motivation for this addressing mode can be explained as
     * follows: say that we have a superconducting quantum interface device
     * (SQUID) based chip. SQUIDs have already been demonstrated passing
     * coherently superposed electrical currents. In a sufficiently
     * quantum-mechanically isolated qubit chip with a classical cache, with
     * both classical RAM and registers likely cryogenically isolated from the
     * environment, SQUIDs could (hopefully) pass coherently superposed
     * electrical currents into the classical RAM cache to load values into a
     * qubit register. The state loaded would be a superposition of the values
     * of all RAM to which coherently superposed electrical currents were
     * passed.
     *
     * In qubit system similar to the MOS 6502, say we have qubit-based
     * "accumulator" and "X index" registers, and say that we start with a
     * superposed X index register. In (classical) X addressing mode, the X
     * index register value acts an offset into RAM from a specified starting
     * address. The X addressing mode of a LoaD Accumulator (LDA) instruction,
     * by the physical mechanism described above, should load the accumulator
     * in quantum parallel with the values of every different address of RAM
     * pointed to in superposition by the X index register. The superposed
     * values in the accumulator are entangled with those in the X index
     * register, by way of whatever values the classical RAM pointed to by X
     * held at the time of the load. (If the RAM at index "36" held an unsigned
     * char value of "27," then the value "36" in the X index register becomes
     * entangled with the value "27" in the accumulator, and so on in quantum
     * parallel for all superposed values of the X index register, at once.) If
     * the X index register or accumulator are then measured, the two registers
     * will both always collapse into a random but valid key-value pair of X
     * index offset and value at that classical RAM address.
     *
     * Note that a "superposed store operation in classical RAM" is not
     * possible by analagous reasoning. Classical RAM would become entangled
     * with both the accumulator and the X register. When the state of the
     * registers was collapsed, we would find that only one "store" operation
     * to a single memory address had actually been carried out, consistent
     * with the address offset in the collapsed X register and the byte value
     * in the collapsed accumulator. It would not be possible by this model to
     * write in quantum parallel to more than one address of classical memory
     * at a time.
     */
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true) = 0;

    /**
     * Add to entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits would usually
     * already be entangled with the "inputStart" bits via a IndexedLDA()
     * operation. With the "inputStart" bits being a "key" and the
     * "outputStart" bits being a value, the permutation state |key, value> is
     * mapped to |key, value + values[key]>. This is similar to classical
     * parallel addition of two arrays.  However, when either of the registers
     * are measured, both registers will collapse into one random VALID
     * key-value pair, with any addition or subtraction done to the "value."
     * See IndexedLDA() for context.
     *
     * FOR BEST EFFICIENCY, the "values" array should be allocated aligned to a 64-byte boundary. (See the unit tests
     * suite code for an example of how to align the allocation.)
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "IndexedADC" and "IndexedSBC" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * IndexedLDA().
     */
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;

    /**
     * Subtract from an entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits would usually
     * already be entangled with the "inputStart" bits via a IndexedLDA()
     * operation.  With the "inputStart" bits being a "key" and the
     * "outputStart" bits being a value, the permutation state |key, value> is
     * mapped to |key, value - values[key]>. This is similar to classical
     * parallel addition of two arrays.  However, when either of the registers
     * are measured, both registers will collapse into one random VALID
     * key-value pair, with any addition or subtraction done to the "value."
     * See QInterface::IndexedLDA for context.
     *
     * FOR BEST EFFICIENCY, the "values" array should be allocated aligned to a 64-byte boundary. (See the unit tests
     * suite code for an example of how to align the allocation.)
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "IndexedADC" and "IndexedSBC" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * IndexedLDA().
     */
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;

    /** Transform a length of qubit register via lookup through a hash table. The hash table must be a one-to-one
     * function, otherwise the behavior of this method is undefined. The value array definition convention is the same
     * as IndexedLDA(). Essentially, this is an IndexedLDA() operation that replaces the index register with the value
     * register, but the lookup table must therefore be one-to-one, for this operation to be unitary, as required. */
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values) = 0;

    /** Swap values of two bits in register */
    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;

    /** Bitwise swap */
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Swap values of two bits in register, and apply phase factor of i if bits are different */
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;

    /** Bitwise swap */
    virtual void ISwap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Square root of Swap gate */
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;

    /** Bitwise square root of swap */
    virtual void SqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Inverse square root of Swap gate */
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;

    /** Bitwise inverse square root of swap */
    virtual void ISqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** The 2-qubit "fSim" gate, (useful in the simulation of particles with fermionic statistics) */
    virtual void FSim(real1 theta, real1 phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;

    /** Bitwise "fSim" */
    virtual void FSim(real1 theta, real1 phi, bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** Reverse all of the bits in a sequence. */
    virtual void Reverse(bitLenInt first, bitLenInt last)
    {
        while ((last > 0) && first < (last - 1)) {
            last--;
            Swap(first, last);
            first++;
        }
    }

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    /**
     * Direct measure of bit probability to be in |1> state
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1 Prob(bitLenInt qubitIndex) = 0;

    /**
     * Direct measure of full permutation probability
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1 ProbAll(bitCapInt fullRegister) = 0;

    /**
     * Direct measure of register permutation probability
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation);

    /**
     * Direct measure of masked permutation probability
     *
     * "mask" masks the bits to check the probability of. "permutation" sets the 0 or 1 value for each bit in the mask.
     * Bits which are set in the mask can be set to 0 or 1 in the permutation, while reset bits in the mask should be 0
     * in the permutation.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation);

    /**
     * Direct measure of masked permutation probability
     *
     * "mask" masks the bits to check the probability of. The probabilities of all permutations of the masked bits, from
     * left/low to right/high are returned in the "probsArray" argument.
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual void ProbMaskAll(const bitCapInt& mask, real1* probsArray);

    /**
     * Statistical measure of masked permutation probability
     *
     * "qPowers" contains powers of 2^n, each representing QInterface bit "n." The order of these values defines a mask
     * for the result bitCapInt, of 2^0 ~ qPowers[0] to 2^(qPowerCount - 1) ~ qPowers[qPowerCount - 1], in contiguous
     * ascending order. "shots" specifies the number of samples to take as if totally re-preparing the pre-measurement
     * state. This method returns a dictionary with keys, which are the (masked-order) measurement results, and values,
     * which are the number of "shots" that produced that particular measurement result. This method does not "collapse"
     * the state of this QInterface. (The idea is to efficiently simulate a potentially statistically random sample of
     * multiple re-preparations of the state right before measurement, and to collect random measurement resutls,
     * without forcing the user to re-prepare or "clone" the state.)
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, const bitLenInt qPowerCount, const unsigned int shots);

    /**
     * Set individual bit to pure |0> (false) or |1> (true) state
     *
     * To set a bit, the bit is first measured. If the result of measurement
     * matches "value," the bit is considered set.  If the result of
     * measurement is the opposite of "value," an X gate is applied to the bit.
     * The state ends up entirely in the "value" state, with a random phase
     * factor.
     */
    virtual void SetBit(bitLenInt qubitIndex1, bool value);

    /**
     * Compare state vectors approximately, component by component, to determine whether this state vector is the same
     * as the target.
     *
     * \warning PSEUDO-QUANTUM
     */

    virtual bool ApproxCompare(QInterfacePtr toCompare) = 0;

    /**
     * Force a calculation of the norm of the state vector, in order to make it unit length before the next probability
     * or measurement operation. (On an actual quantum computer, the state should never require manual normalization.)
     *
     * \warning PSEUDO-QUANTUM
     */

    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG) = 0;

    /**
     * Apply the normalization factor found by UpdateRunningNorm() or on the fly by a single bit gate. (On an actual
     * quantum computer, the state should never require manual normalization.)
     *
     * \warning PSEUDO-QUANTUM
     */

    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG) = 0;

    /**
     * If asynchronous work is still running, block until it finishes. Note that this is never necessary to get correct,
     * timely return values. QEngines and other layers will always internally "Finish" when necessary for correct return
     * values. This is primarily for debugging and benchmarking.
     */

    virtual void Finish(){};

    /**
     * Returns "false" if asynchronous work is still running, and "true" if all previously dispatched asynchronous work
     * is done.
     */

    virtual bool isFinished() { return true; };

    /**
     *  Qrack::QUnit types maintain explicit separation of representations of qubits, which reduces memory usage and
     * increases gate speed. This method is used to manually attempt internal separation of a QUnit subsytem. We attempt
     * a Decompose() operation, on a state which might not be separable. If the state is not separable, we abort and
     * return false. Otherwise, we complete the operation, add the separated subsystem back in place into the QUnit
     * "shards," and return true.
     *
     * \warning PSEUDO-QUANTUM
     *
     * This should never change the logical/physical state of the QInterface, only possibly its internal representation,
     * for simulation optimization purposes. This is not a truly quantum computational operation, but it also does not
     * lead to nonphysical effects.
     */
    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1) { return false; }

    /**
     *  Clone this QInterface
     */
    virtual QInterfacePtr Clone() = 0;

    /** @} */
};
} // namespace Qrack
