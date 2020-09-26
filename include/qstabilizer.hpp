//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
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

#include <cstdint>

#include "common/qrack_types.hpp"
#include "common/rdrandwrapper.hpp"

namespace Qrack {

class QStabilizer;
typedef std::shared_ptr<QStabilizer> QStabilizerPtr;

class QStabilizer {
protected:
    // # of qubits
    bitLenInt qubitCount;
    // (2n+1)*n matrix for stabilizer/destabilizer x bits (there's one "scratch row" at the bottom)
    std::vector<std::vector<bool>> x;
    // (2n+1)*n matrix for z bits
    std::vector<std::vector<bool>> z;
    // Phase bits: 0 for +1, 1 for i, 2 for -1, 3 for -i.  Normally either 0 or 2.
    std::vector<uint8_t> r;

    uint32_t randomSeed;
    qrack_rand_gen_ptr rand_generator;
    std::uniform_real_distribution<real1> rand_distribution;
    std::shared_ptr<RdRandom> hardware_rand_generator;

    bitCapInt pow2(const bitLenInt& qubit) { return ONE_BCI << (bitCapInt)qubit; }

public:
    QStabilizer(const bitLenInt& n, const bitCapInt& perm = 0, const bool& useHardwareRNG = true,
        qrack_rand_gen_ptr rgp = nullptr);

    bitLenInt GetQubitCount() { return qubitCount; }

    bitLenInt GetMaxQPower() { return pow2(qubitCount); }

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
            return hardware_rand_generator->Next() < (ONE_R1 / 2U);
        } else {
            return rand_distribution(*rand_generator) < (ONE_R1 / 2U);
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

    /// Returns the result of applying the Pauli operator in the "scratch space" of q to |0...0>
    void setBasisState(const real1& nrm, complex* stateVec);

    void DecomposeDispose(const bitLenInt& start, const bitLenInt& length, QStabilizerPtr toCopy);

public:
    /// Apply a CNOT gate with control and target
    void CNOT(const bitLenInt& control, const bitLenInt& target);
    /// Apply a Hadamard gate to target
    void H(const bitLenInt& target);
    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    void S(const bitLenInt& target);

    // TODO: Custom implementations for decompositions:
    virtual void Z(const bitLenInt& target)
    {
        S(target);
        S(target);
    }

    virtual void IS(const bitLenInt& target)
    {
        Z(target);
        S(target);
    }

    virtual void X(const bitLenInt& target)
    {
        H(target);
        Z(target);
        H(target);
    }

    virtual void Y(const bitLenInt& target)
    {
        IS(target);
        X(target);
        S(target);
    }

    virtual void CZ(const bitLenInt& control, const bitLenInt& target)
    {
        H(target);
        CNOT(control, target);
        H(target);
    }

    virtual void Swap(const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        CNOT(qubit1, qubit2);
        CNOT(qubit2, qubit1);
        CNOT(qubit1, qubit2);
    }

    virtual void ISwap(const bitLenInt& qubit1, const bitLenInt& qubit2)
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
    bool M(const bitLenInt& t, bool result = false, const bool& doForce = false, const bool& doApply = true);

    /// Convert the state to ket notation
    void GetQuantumState(complex* stateVec);

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

    bitLenInt Compose(QStabilizerPtr toCopy)
    {
        bitLenInt toRet = qubitCount;
        Compose(toCopy, qubitCount);
        return toRet;
    }
    bitLenInt Compose(QStabilizerPtr toCopy, const bitLenInt& start);
    void Decompose(const bitLenInt& start, QStabilizerPtr destination)
    {
        DecomposeDispose(start, destination->GetQubitCount(), destination);
    }

    void Dispose(const bitLenInt& start, const bitLenInt& length)
    {
        DecomposeDispose(start, length, (QStabilizerPtr)NULL);
    }

    bool ApproxCompare(QStabilizerPtr o);
};
} // namespace Qrack
