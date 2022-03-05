//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// Adapted from:
//
// CHP: CNOT-Hadamard-Phase
// Stabilizer Quantum Computer Simulator
// by Scott Aaronson
// Last modified June 30, 2004
//
// Thanks to Simon Anders and Andrew Cross for bugfixes
//
// https://www.scottaaronson.com/chp/
//
// Daniel Strano and the Qrack contributers appreciate Scott Aaronson's open sharing of the CHP code, and we hope that
// vm6502q/qrack is one satisfactory framework by which CHP could be adapted to enter the C++ STL. Our project
// philosophy aims to raise the floor of decentralized quantum computing technology access across all modern platforms,
// for all people, not commercialization.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"

#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
#include "common/dispatchqueue.hpp"
#endif

#include <cstdint>

namespace Qrack {

struct AmplitudeEntry {
    bitCapIntOcl permutation;
    complex amplitude;

    AmplitudeEntry(const bitCapInt& p, const complex& a)
        : permutation(p)
        , amplitude(a)
    {
    }
};

class QStabilizer;
typedef std::shared_ptr<QStabilizer> QStabilizerPtr;

class QStabilizer : public QInterface {
protected:
    // (2n+1)*n matrix for stabilizer/destabilizer x bits (there's one "scratch row" at the bottom)
    std::vector<std::vector<bool>> x;
    // (2n+1)*n matrix for z bits
    std::vector<std::vector<bool>> z;
    // Phase bits: 0 for +1, 1 for i, 2 for -1, 3 for -i.  Normally either 0 or 2.
    std::vector<uint8_t> r;

    uint32_t randomSeed;
    qrack_rand_gen_ptr rand_generator;
#if defined(_WIN32) && !defined(__CYGWIN__)
    std::uniform_int_distribution<short> rand_distribution;
#else
    std::uniform_int_distribution<char> rand_distribution;
#endif
    std::shared_ptr<RdRandom> hardware_rand_generator;

    unsigned rawRandBools;
    unsigned rawRandBoolsRemaining;

#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
    DispatchQueue dispatchQueue;
    bitLenInt dispatchThreshold;
#endif

    typedef std::function<void(const bitLenInt&)> StabilizerParallelFunc;
    typedef std::function<void(void)> DispatchFn;
    void Dispatch(DispatchFn fn)
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        if (qubitCount >= dispatchThreshold) {
            dispatchQueue.dispatch(fn);
        } else {
            Finish();
            fn();
        }
#else
        fn();
#endif
    }

    void ParFor(StabilizerParallelFunc fn)
    {
        Dispatch([this, fn] {
            const bitLenInt maxLcv = qubitCount << 1U;
            for (bitLenInt i = 0; i < maxLcv; i++) {
                fn(i);
            }
        });
    }

public:
    QStabilizer(bitLenInt n, bitCapInt perm = 0, qrack_rand_gen_ptr rgp = nullptr, bool useHardwareRNG = true,
        bool randomGlobalPhase = true);

    QInterfacePtr Clone()
    {
        Finish();

        QStabilizerPtr clone =
            std::make_shared<QStabilizer>(qubitCount, 0, hardware_rand_generator != NULL, rand_generator);
        clone->Finish();

        clone->x = x;
        clone->z = z;
        clone->r = r;
        clone->randomSeed = randomSeed;

        return clone;
    }

    virtual ~QStabilizer() { Dump(); }

    void Finish()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.finish();
#endif
    };

    bool isFinished()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        return dispatchQueue.isFinished();
#else
        return true;
#endif
    }

    void Dump()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.dump();
#endif
    }

    bitLenInt GetQubitCount() { return qubitCount; }

    bitCapInt GetMaxQPower() { return pow2(qubitCount); }

    void SetPermutation(const bitCapInt& perm);

    void SetRandomSeed(uint32_t seed)
    {
        if (rand_generator != NULL) {
            rand_generator->seed(seed);
        }
    }

    bool Rand()
    {
        if (hardware_rand_generator != NULL) {
            if (!rawRandBoolsRemaining) {
                rawRandBools = hardware_rand_generator->NextRaw();
                rawRandBoolsRemaining = sizeof(unsigned) * bitsInByte;
            }
            rawRandBoolsRemaining--;

            return (bool)((rawRandBools >> rawRandBoolsRemaining) & 1U);
        } else {
            return (bool)rand_distribution(*rand_generator);
        }
    }

protected:
    /// Sets row i equal to row k
    void rowcopy(const bitLenInt& i, const bitLenInt& k);
    /// Swaps row i and row k
    void rowswap(const bitLenInt& i, const bitLenInt& k);
    /// Sets row i equal to the bth observable (X_1,...X_n,Z_1,...,Z_n)
    void rowset(const bitLenInt& i, bitLenInt b);
    /// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k
    uint8_t clifford(const bitLenInt& i, const bitLenInt& k);
    /// Left-multiply row i by row k
    void rowmult(const bitLenInt& i, const bitLenInt& k);

    /**
     * Do Gaussian elimination to put the stabilizer generators in the following form:
     * At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
     * (Return value = number of such generators = log_2 of number of nonzero basis states)
     * At the bottom, generators containing Z's only in quasi-upper-triangular form.
     */
    bitLenInt gaussian();

    /**
     * Finds a Pauli operator P such that the basis state P|0...0> occurs with nonzero amplitude in q, and
     * writes P to the scratch space of q.  For this to work, Gaussian elimination must already have been
     * performed on q.  g is the return value from gaussian(q).
     */
    void seed(const bitLenInt& g);

    /// Helper for setBasisState() and setBasisProb()
    AmplitudeEntry getBasisAmp(const real1_f& nrm);

    /// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisState(const real1_f& nrm, complex* stateVec, QInterfacePtr eng);

    /// Returns the probability from applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisProb(const real1_f& nrm, real1* outputProbs);

    void DecomposeDispose(const bitLenInt start, const bitLenInt length, QStabilizerPtr toCopy);

public:
    /// Apply a CNOT gate with control and target
    void CNOT(const bitLenInt& control, const bitLenInt& target);
    /// Apply a CY gate with control and target
    void CY(const bitLenInt& control, const bitLenInt& target);
    /// Apply a CZ gate with control and target
    void CZ(const bitLenInt& control, const bitLenInt& target);
    /// Apply a Hadamard gate to target
    void H(const bitLenInt& target);
    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    void S(const bitLenInt& target);
    /// Apply an inverse phase gate (|0>->|0>, |1>->-i|1>, or "S adjoint") to qubit b
    void IS(const bitLenInt& target);
    /// Apply a phase gate (|0>->|0>, |1>->-|1>, or "Z") to qubit b
    void Z(const bitLenInt& target);
    /// Apply an X (or NOT) gate to target
    void X(const bitLenInt& target);
    /// Apply a Pauli Y gate to target
    void Y(const bitLenInt& target);
    /// Apply square root of X gate
    void SqrtX(const bitLenInt& target);
    /// Apply inverse square root of X gate
    void ISqrtX(const bitLenInt& target);
    /// Apply square root of Y gate
    void SqrtY(const bitLenInt& target);
    /// Apply inverse square root of Y gate
    void ISqrtY(const bitLenInt& target);

    void Swap(const bitLenInt& qubit1, const bitLenInt& qubit2);

    void ISwap(const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        S(qubit1);
        S(qubit2);
        H(qubit1);
        CNOT(qubit1, qubit2);
        CNOT(qubit2, qubit1);
        H(qubit2);
    }

    /**
     * Measure qubit b
     */
    bool ForceM(bitLenInt t, bool result, bool doForce = true, bool doApply = true);

    /// Convert the state to ket notation
    void GetQuantumState(complex* stateVec);

    /// Convert the state to ket notation, directly into another QInterface
    void GetQuantumState(QInterfacePtr eng);

    /// Get all probabilities corresponding to ket notation
    void GetProbs(real1* outputProbs);

    /**
     * Returns "true" if target qubit is a Z basis eigenstate
     */
    bool IsSeparableZ(const bitLenInt& target);
    /**
     * Returns "true" if target qubit is an X basis eigenstate
     */
    bool IsSeparableX(const bitLenInt& target);
    /**
     * Returns "true" if target qubit is a Y basis eigenstate
     */
    bool IsSeparableY(const bitLenInt& target);
    /**
     * Returns:
     * 0 if target qubit is not separable
     * 1 if target qubit is a Z basis eigenstate
     * 2 if target qubit is an X basis eigenstate
     * 3 if target qubit is a Y basis eigenstate
     */
    uint8_t IsSeparable(const bitLenInt& target);

    bitLenInt Compose(QStabilizerPtr toCopy) { return Compose(toCopy, qubitCount); }
    bitLenInt Compose(QStabilizerPtr toCopy, bitLenInt start);
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        DecomposeDispose(start, dest->GetQubitCount(), std::dynamic_pointer_cast<QStabilizer>(dest));
    }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QStabilizerPtr)NULL); }
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt ignored)
    {
        DecomposeDispose(start, length, (QStabilizerPtr)NULL);
    }
    bool CanDecomposeDispose(const bitLenInt start, const bitLenInt length);

    bool ApproxCompare(QStabilizerPtr o);

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1)
    {
        // Intentionally left blank
    }

    void Mtrx(const complex* mtrx, bitLenInt target);
    void Phase(complex topLeft, complex bottomRight, bitLenInt target);
    void Invert(complex topRight, complex bottomLeft, bitLenInt target);
    void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);

    void SetQuantumState(const complex* inputState)
    {
        throw std::logic_error("QStabilizer::SetQuantumState() not implemented!");
    }
    complex GetAmplitude(bitCapInt perm) { throw std::logic_error("QStabilizer::GetAmplitude() not implemented!"); }
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        throw std::logic_error("QStabilizer::SetAmplitude() not implemented!");
    }
};
} // namespace Qrack
