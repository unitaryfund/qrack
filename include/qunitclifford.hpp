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

#include "qstabilizer.hpp"

namespace Qrack {

class QUnitClifford;
typedef std::shared_ptr<QUnitClifford> QUnitCliffordPtr;

struct CliffordShard {
    bitLenInt mapped;
    QStabilizerPtr unit;

    CliffordShard(bitLenInt m = 0U, QStabilizerPtr u = NULL)
        : mapped(m)
        , unit(u)
    {
        // Intentionally left blank
    }

    CliffordShard(const CliffordShard& o)
        : mapped(o.mapped)
        , unit(o.unit)
    {
        // Intentionally left blank
    }
};

class QUnitClifford : public QInterface {
protected:
    std::vector<CliffordShard> shards;

    struct QSortEntry {
        bitLenInt bit;
        bitLenInt mapped;
        bool operator<(const QSortEntry& rhs) { return mapped < rhs.mapped; }
        bool operator>(const QSortEntry& rhs) { return mapped > rhs.mapped; }
    };
    void SortUnit(QStabilizerPtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high);

    void Detach(bitLenInt start, bitLenInt length, QUnitCliffordPtr dest);

    QStabilizerPtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    QStabilizerPtr EntangleAll()
    {
        std::vector<bitLenInt> bits(qubitCount);
        std::vector<bitLenInt*> ebits(qubitCount);
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            bits[i] = i;
            ebits[i] = &bits[i];
        }

        QStabilizerPtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
        OrderContiguous(toRet);

        return toRet;
    }

    void OrderContiguous(QStabilizerPtr unit);

    typedef std::function<void(QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx)>
        CGateFn;
    void CGate(bitLenInt control, bitLenInt target, const complex* mtrx, CGateFn fn)
    {
        if (shards[control].unit == shards[target].unit) {
            fn(shards[control].unit, shards[control].mapped, shards[target].mapped, mtrx);
            TrySeparate(control);

            TrySeparate(target);
            return;
        }
        std::vector<bitLenInt> bits{ control, target };
        std::vector<bitLenInt*> ebits{ &bits[0U], &bits[1U] };
        QStabilizerPtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());
        fn(unit, bits[0U], bits[1U], mtrx);
    }

    QInterfacePtr CloneBody(QUnitCliffordPtr copyPtr);

    bool SeparateBit(bool value, bitLenInt qubit);

public:
    QUnitClifford(bitLenInt n, bitCapInt perm = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex ignored = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true, bool ignored2 = false,
        int64_t ignored3 = -1, bool useHardwareRNG = true, bool ignored4 = false, real1_f ignored5 = REAL1_EPSILON,
        std::vector<int64_t> ignored6 = {}, bitLenInt ignored7 = 0U, real1_f ignored8 = FP_NORM_EPSILON_F);

    ~QUnitClifford() { Dump(); }

    QInterfacePtr Clone();

    bool isClifford() { return true; };
    bool isClifford(bitLenInt qubit) { return true; };

    bitLenInt GetQubitCount() { return qubitCount; }

    bitCapInt GetMaxQPower() { return pow2(qubitCount); }

    void SetDevice(int64_t dID) {}

    bitLenInt PermCount()
    {
        QUnitCliffordPtr thisCopy = std::dynamic_pointer_cast<QUnitClifford>(Clone());
        thisCopy->EntangleAll();
        return thisCopy->shards[0U].unit->gaussian();
    }

    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    QStabilizerPtr MakeStabilizer(bitLenInt length = 1U, bitCapInt perm = 0U);

    void SetQuantumState(const complex* inputState);
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        throw std::domain_error("QStabilizer::SetAmplitude() not implemented!");
    }

    /// Apply a CNOT gate with control and target
    void CNOT(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->CNOT(c, t);
        });
    }
    /// Apply a CY gate with control and target
    void CY(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) { unit->CY(c, t); });
    }
    /// Apply a CZ gate with control and target
    void CZ(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL,
            [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) { unit->CZ(c, t); });
    }
    /// Apply an (anti-)CNOT gate with control and target
    void AntiCNOT(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->AntiCNOT(c, t);
        });
    }
    /// Apply an (anti-)CY gate with control and target
    void AntiCY(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->AntiCY(c, t);
        });
    }
    /// Apply an (anti-)CZ gate with control and target
    void AntiCZ(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->AntiCZ(c, t);
        });
    }
    /// Apply a Hadamard gate to target
    using QInterface::H;
    void H(bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::H qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->H(shard.mapped);
    }
    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    void S(bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::S qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->S(shard.mapped);
    }
    /// Apply an inverse phase gate (|0>->|0>, |1>->-i|1>, or "S adjoint") to qubit b
    void IS(bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::IS qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->IS(shard.mapped);
    }
    /// Apply a phase gate (|0>->|0>, |1>->-|1>, or "Z") to qubit b
    void Z(bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Z qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->Z(shard.mapped);
    }
    /// Apply an X (or NOT) gate to target
    using QInterface::X;
    void X(bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::X qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->X(shard.mapped);
    }
    /// Apply a Pauli Y gate to target
    void Y(bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Y qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->Y(shard.mapped);
    }
    // Swap two bits
    void Swap(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Swap qubit index parameter must be within allocated qubit bounds!");
        }

        if (qubit2 >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Swap qubit index parameter must be within allocated qubit bounds!");
        }

        if (qubit1 == qubit2) {
            return;
        }

        // Simply swap the bit mapping.
        std::swap(shards[qubit1], shards[qubit2]);
    }
    // Swap two bits and apply a phase factor of i if they are different
    void ISwap(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->ISwap(c, t);
        });
    }
    // Swap two bits and apply a phase factor of -i if they are different
    void IISwap(bitLenInt c, bitLenInt t)
    {
        CGate(c, t, NULL, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* unused) {
            unit->IISwap(c, t);
        });
    }

    /// Measure qubit t
    bool ForceM(bitLenInt t, bool result, bool doForce = true, bool doApply = true);

    /// Measure all qubits
    bitCapInt MAll()
    {
        bitCapInt toRet = QInterface::MAll();
        SetPermutation(toRet);
        return toRet;
    }

    std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);

    void MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);

    /// Convert the state to ket notation
    void GetQuantumState(complex* stateVec);

    /// Convert the state to ket notation, directly into another QInterface
    void GetQuantumState(QInterfacePtr eng);

    /// Convert the state to sparse ket notation
    std::map<bitCapInt, complex> GetQuantumState();

    /// Get all probabilities corresponding to ket notation
    void GetProbs(real1* outputProbs);

    /// Get a single basis state amplitude
    complex GetAmplitude(bitCapInt perm);

    /// Get a single basis state amplitude
    std::vector<complex> GetAmplitudes(std::vector<bitCapInt> perms);

    /**
     * Returns "true" if target qubit is a Z basis eigenstate
     */
    bool IsSeparableZ(const bitLenInt& t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument("QUnitClifford::IsSeparableZ qubit index is out-of-bounds!");
        }
        CliffordShard& shard = shards[t];

        return shard.unit->IsSeparableZ(shard.mapped);
    }

    /**
     * Returns "true" if target qubit is an X basis eigenstate
     */
    bool IsSeparableX(const bitLenInt& t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument("QUnitClifford::IsSeparableX qubit index is out-of-bounds!");
        }
        CliffordShard& shard = shards[t];

        return shard.unit->IsSeparableX(shard.mapped);
    }
    /**
     * Returns "true" if target qubit is a Y basis eigenstate
     */
    bool IsSeparableY(const bitLenInt& t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument("QUnitClifford::IsSeparableY qubit index is out-of-bounds!");
        }
        CliffordShard& shard = shards[t];

        return shard.unit->IsSeparableY(shard.mapped);
    }
    /**
     * Returns:
     * 0 if target qubit is not separable
     * 1 if target qubit is a Z basis eigenstate
     * 2 if target qubit is an X basis eigenstate
     * 3 if target qubit is a Y basis eigenstate
     */
    uint8_t IsSeparable(const bitLenInt& t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument("QUnitClifford::IsSeparable qubit index is out-of-bounds!");
        }
        CliffordShard& shard = shards[t];

        return shard.unit->IsSeparable(shard.mapped);
    }

    using QInterface::Compose;
    bitLenInt Compose(QUnitCliffordPtr toCopy) { return Compose(toCopy, qubitCount); }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QUnitClifford>(toCopy)); }
    bitLenInt Compose(QUnitCliffordPtr toCopy, bitLenInt start)
    {
        if (start > qubitCount) {
            throw std::invalid_argument("QUnit::Compose start index is out-of-bounds!");
        }

        /* Create a clone of the quantum state in toCopy. */
        QUnitCliffordPtr clone = std::dynamic_pointer_cast<QUnitClifford>(toCopy->Clone());

        /* Insert the new shards in the middle */
        shards.insert(shards.begin() + start, clone->shards.begin(), clone->shards.end());

        SetQubitCount(qubitCount + toCopy->GetQubitCount());

        return start;
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QUnitClifford>(toCopy), start);
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QUnitClifford>(dest));
    }
    void Decompose(bitLenInt start, QUnitCliffordPtr dest) { Detach(start, dest->GetQubitCount(), dest); }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        QUnitCliffordPtr dest = std::make_shared<QUnitClifford>(
            length, 0U, rand_generator, CMPLX_DEFAULT_ARG, doNormalize, randGlobalPhase, false, 0U, useRDRAND);

        Decompose(start, dest);

        return dest;
    }
    void Dispose(bitLenInt start, bitLenInt length) { Detach(start, length, nullptr); }
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm) { Detach(start, length, nullptr); }
    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (!length) {
            return start;
        }

        QUnitCliffordPtr nQubits = std::make_shared<QUnitClifford>(length, 0U, rand_generator, CMPLX_DEFAULT_ARG, false,
            randGlobalPhase, false, -1, hardware_rand_generator != NULL);
        return Compose(nQubits, start);
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        // Intentionally left blank
    }
    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QUnitClifford>(toCompare));
    }
    virtual real1_f SumSqrDiff(QUnitCliffordPtr toCompare);

    real1_f Prob(bitLenInt qubit)
    {
        if (qubit >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Prob qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[qubit];
        return shard.unit->Prob(shard.mapped);
    }

    void Mtrx(const complex* mtrx, bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Mtrx qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->Mtrx(mtrx, shard.mapped);
    }
    void Phase(complex topLeft, complex bottomRight, bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Phase qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->Phase(topLeft, bottomRight, shard.mapped);
    }
    void Invert(complex topRight, complex bottomLeft, bitLenInt t)
    {
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::Invert qubit index parameter must be within allocated qubit bounds!");
        }
        CliffordShard& shard = shards[t];
        shard.unit->Invert(topRight, bottomLeft, shard.mapped);
    }
    void MCPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt t)
    {
        if (!controls.size()) {
            Phase(topLeft, bottomRight, t);
            return;
        }
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MCPhase target qubit index parameter must be within allocated qubit bounds!");
        }
        if (controls.size() > 1U) {
            throw std::invalid_argument("QUnitClifford::MCPhase can only have one control qubit!");
        }
        const bitLenInt c = controls[0U];
        if (c >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MCPhase control qubit index parameter must be within allocated qubit bounds!");
        }

        const complex mtrx[4]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MCPhase({ c }, mtrx[0U], mtrx[3U], t);
        });
    }
    void MACPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt t)
    {
        if (!controls.size()) {
            Phase(topLeft, bottomRight, t);
            return;
        }
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MACPhase target qubit index parameter must be within allocated qubit bounds!");
        }
        if (controls.size() > 1U) {
            throw std::invalid_argument("QUnitClifford::MACPhase can only have one control qubit!");
        }
        const bitLenInt c = controls[0U];
        if (c >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MACPhase control qubit index parameter must be within allocated qubit bounds!");
        }

        const complex mtrx[4]{ topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MACPhase({ c }, mtrx[0U], mtrx[3U], t);
        });
    }
    void MCInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt t)
    {
        if (!controls.size()) {
            Invert(topRight, bottomLeft, t);
            return;
        }
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MCInvert target qubit index parameter must be within allocated qubit bounds!");
        }
        if (controls.size() > 1U) {
            throw std::invalid_argument("QUnitClifford::MCInvert can only have one control qubit!");
        }
        const bitLenInt c = controls[0U];
        if (c >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MCInvert control qubit index parameter must be within allocated qubit bounds!");
        }

        const complex mtrx[4]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MCInvert({ c }, mtrx[1U], mtrx[2U], t);
        });
    }
    void MACInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt t)
    {
        if (!controls.size()) {
            Invert(topRight, bottomLeft, t);
            return;
        }
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MACInvert target qubit index parameter must be within allocated qubit bounds!");
        }
        if (controls.size() > 1U) {
            throw std::invalid_argument("QUnitClifford::MACInvert can only have one control qubit!");
        }
        const bitLenInt c = controls[0U];
        if (c >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MACInvert control qubit index parameter must be within allocated qubit bounds!");
        }

        const complex mtrx[4]{ ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MACInvert({ c }, mtrx[1U], mtrx[2U], t);
        });
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt t)
    {
        if (!controls.size()) {
            Mtrx(mtrx, t);
            return;
        }
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MCMtrx target qubit index parameter must be within allocated qubit bounds!");
        }
        if (controls.size() > 1U) {
            throw std::invalid_argument("QUnitClifford::MCMtrx can only have one control qubit!");
        }
        const bitLenInt c = controls[0U];
        if (c >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MCMtrx control qubit index parameter must be within allocated qubit bounds!");
        }

        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MCMtrx({ c }, mtrx, t);
        });
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt t)
    {
        if (!controls.size()) {
            Mtrx(mtrx, t);
            return;
        }
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MACMtrx target qubit index parameter must be within allocated qubit bounds!");
        }
        if (controls.size() > 1U) {
            throw std::invalid_argument("QUnitClifford::MACMtrx can only have one control qubit!");
        }
        const bitLenInt c = controls[0U];
        if (c >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::MACMtrx control qubit index parameter must be within allocated qubit bounds!");
        }

        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->MACMtrx({ c }, mtrx, t);
        });
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt c, bitLenInt t)
    {
        if (c >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::FSim control qubit index parameter must be within allocated qubit bounds!");
        }
        if (t >= qubitCount) {
            throw std::invalid_argument(
                "QUnitClifford::FSim target qubit index parameter must be within allocated qubit bounds!");
        }

        const complex mtrx[4]{ (real1)theta, (real1)phi, ZERO_CMPLX, ZERO_CMPLX };
        CGate(c, t, mtrx, [](QStabilizerPtr unit, const bitLenInt& c, const bitLenInt& t, const complex* mtrx) {
            unit->FSim((real1_f)std::real(mtrx[0U]), (real1_f)std::real(mtrx[1U]), c, t);
        });
    }

    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f ignored)
    {
        for (size_t i = 0U; i < qubits.size(); ++i) {
            if (!TrySeparate(qubits[i])) {
                return false;
            }
        }

        return true;
    }
    bool TrySeparate(bitLenInt qubit)
    {
        CliffordShard& shard = shards[qubit];

        if (shard.unit->GetQubitCount() <= 1U) {
            return true;
        }

        if (!shard.unit->TrySeparate(shard.mapped)) {
            return false;
        }

        // If TrySeparate() == true, this bit can be decomposed.
        QStabilizerPtr sepUnit = std::dynamic_pointer_cast<QStabilizer>(shard.unit->Decompose(shard.mapped, 1U));

        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            if ((shard.unit == shards[i].unit) && (shard.mapped < shards[i].mapped)) {
                --(shards[i].mapped);
            }
        }

        shard.mapped = 0U;
        shard.unit = sepUnit;

        return true;
    }
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2) { return TrySeparate(qubit1) && TrySeparate(qubit2); }

    friend std::ostream& operator<<(std::ostream& os, const QUnitCliffordPtr s);
    friend std::istream& operator>>(std::istream& is, const QUnitCliffordPtr s);
};
} // namespace Qrack
