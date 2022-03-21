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
    typedef std::vector<bool> BoolVector;
    // (2n+1)*n matrix for stabilizer/destabilizer x bits (there's one "scratch row" at the bottom)
    std::vector<BoolVector> x;
    // (2n+1)*n matrix for z bits
    std::vector<BoolVector> z;
    // Phase bits: 0 for +1, 1 for i, 2 for -1, 3 for -i.  Normally either 0 or 2.
    std::vector<uint8_t> r;
    complex phaseOffset;

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

    bool TrimControls(const bitLenInt* lControls, bitLenInt lControlLen, std::vector<bitLenInt>& output)
    {
        for (bitLenInt i = 0; i < lControlLen; i++) {
            const bitLenInt bit = lControls[i];
            if (!IsSeparableZ(bit)) {
                output.push_back(bit);
                continue;
            }
            if (!M(bit)) {
                return true;
            }
        }

        return false;
    }

public:
    QStabilizer(bitLenInt n, bitCapInt perm = 0, qrack_rand_gen_ptr rgp = nullptr, complex ignored = CMPLX_DEFAULT_ARG,
        bool doNorm = false, bool randomGlobalPhase = true, bool ignored2 = false, int ignored3 = -1,
        bool useHardwareRNG = true, bool ignored4 = false, real1_f ignored5 = REAL1_EPSILON,
        std::vector<int> ignored6 = {}, bitLenInt ignored7 = 0, real1_f ignored8 = FP_NORM_EPSILON);

    QInterfacePtr Clone()
    {
        Finish();

        QStabilizerPtr clone = std::make_shared<QStabilizer>(qubitCount, 0, rand_generator, CMPLX_DEFAULT_ARG, false,
            randGlobalPhase, false, -1, hardware_rand_generator != NULL);
        clone->Finish();

        clone->x = x;
        clone->z = z;
        clone->r = r;
        clone->phaseOffset = phaseOffset;
        clone->randomSeed = randomSeed;

        return clone;
    }

    virtual ~QStabilizer() { Dump(); }

    virtual bool isClifford() { return true; };
    virtual bool isClifford(bitLenInt qubit) { return true; };

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

    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

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

    real1_f ApproxCompareHelper(
        QStabilizerPtr toCompare, bool isDiscreteBool, real1_f error_tol = TRYDECOMPOSE_EPSILON);

public:
    void SetQuantumState(const complex* inputState)
    {
        if (qubitCount > 1U) {
            throw std::domain_error("QStabilizer::SetQuantumState() not generally implemented!");
        }

        SetPermutation(0);

        const real1 prob = (real1)clampProb(norm(inputState[1]));
        const real1 sqrtProb = sqrt(prob);
        const real1 sqrt1MinProb = (real1)sqrt(clampProb(ONE_R1 - prob));
        const complex phase0 = std::polar(ONE_R1, arg(inputState[0]));
        const complex phase1 = std::polar(ONE_R1, arg(inputState[1]));
        const complex mtrx[4] = { sqrt1MinProb * phase0, sqrtProb * phase0, sqrtProb * phase1, -sqrt1MinProb * phase1 };
        Mtrx(mtrx, 0);
    }
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        throw std::domain_error("QStabilizer::SetAmplitude() not implemented!");
    }

    /// Apply a CNOT gate with control and target
    virtual void CNOT(bitLenInt control, bitLenInt target);
    /// Apply a CY gate with control and target
    virtual void CY(bitLenInt control, bitLenInt target);
    /// Apply a CZ gate with control and target
    virtual void CZ(bitLenInt control, bitLenInt target);
    /// Apply a Hadamard gate to target
    virtual void H(bitLenInt qubitIndex);
    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    virtual void S(bitLenInt qubitIndex);
    /// Apply an inverse phase gate (|0>->|0>, |1>->-i|1>, or "S adjoint") to qubit b
    virtual void IS(bitLenInt qubitIndex);
    /// Apply a phase gate (|0>->|0>, |1>->-|1>, or "Z") to qubit b
    virtual void Z(bitLenInt qubitIndex);
    /// Apply an X (or NOT) gate to target
    virtual void X(bitLenInt qubitIndex);
    /// Apply a Pauli Y gate to target
    virtual void Y(bitLenInt qubitIndex);

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    /// Measure qubit t
    virtual bool ForceM(bitLenInt t, bool result, bool doForce = true, bool doApply = true);

    /// Measure all qubits
    virtual bitCapInt MAll()
    {
        bitCapInt toRet = QInterface::MAll();
        SetPermutation(toRet);
        return toRet;
    }

    /// Get the phase radians of the lowest permutation nonzero amplitude
    virtual real1_f FirstNonzeroPhase();

    /// Convert the state to ket notation
    virtual void GetQuantumState(complex* stateVec);

    /// Convert the state to ket notation, directly into another QInterface
    virtual void GetQuantumState(QInterfacePtr eng);

    /// Get all probabilities corresponding to ket notation
    virtual void GetProbs(real1* outputProbs);

    /// Get a single basis state amplitude
    virtual complex GetAmplitude(bitCapInt perm);

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

    using QInterface::Compose;
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QStabilizer>(toCopy)); }
    virtual bitLenInt Compose(QStabilizerPtr toCopy) { return Compose(toCopy, qubitCount); }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizer>(toCopy), start);
    }
    virtual bitLenInt Compose(QStabilizerPtr toCopy, bitLenInt start);
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        DecomposeDispose(start, dest->GetQubitCount(), std::dynamic_pointer_cast<QStabilizer>(dest));
    }
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QStabilizerPtr)NULL); }
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt ignored)
    {
        DecomposeDispose(start, length, (QStabilizerPtr)NULL);
    }
    bool CanDecomposeDispose(const bitLenInt start, const bitLenInt length);

    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1)
    {
        if (!randGlobalPhase) {
            phaseOffset *= std::polar(ONE_R1, (real1)phaseArg);
        }
    }
    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizer>(toCompare), false);
    }
    virtual bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return error_tol >= ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizer>(toCompare), true, error_tol);
    }

    virtual real1_f Prob(bitLenInt qubit);

    virtual void Mtrx(const complex* mtrx, bitLenInt target);
    virtual void Phase(complex topLeft, complex bottomRight, bitLenInt target);
    virtual void Invert(complex topRight, complex bottomLeft, bitLenInt target);
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    virtual void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2);

    virtual bool TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f ignored)
    {
        for (bitLenInt i = 0U; i < length; i++) {
            Swap(qubits[i], i);
        }

        const bool toRet = CanDecomposeDispose(0U, 2U);

        for (bitLenInt i = 0U; i < length; i++) {
            Swap(qubits[i], i);
        }

        return toRet;
    }
    virtual bool TrySeparate(bitLenInt qubit) { return CanDecomposeDispose(qubit, 1U); }
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
    {
        Swap(qubit1, 0U);
        Swap(qubit2, 1U);

        const bool toRet = CanDecomposeDispose(0U, 2U);

        Swap(qubit1, 0U);
        Swap(qubit2, 1U);

        return toRet;
    }
};
} // namespace Qrack
