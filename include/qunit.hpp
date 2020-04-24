//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <cfloat>
#include <random>

#include "qinterface.hpp"

namespace Qrack {

// "PhaseShard" optimizations are basically just a very specific "gate fusion" type optimization, where multiple gates
// are composed into single product gates before application to the state vector, to reduce the total number of gates
// that need to be applied. Rather than handling this as a "QFusion" layer optimization, which will typically sit
// BETWEEN a base QEngine set of "shards" and a QUnit that owns them, this particular gate fusion optimization can be
// avoid representational entanglement in QUnit in the first place, which QFusion would not help with. Alternatively,
// another QFusion would have to be in place ABOVE the QUnit layer, (with QEngine "below,") for this to work.
// Additionally, QFusion is designed to handle more general gate fusion, not specifically controlled phase gates, which
// are entirely commuting among each other and possibly a jumping-off point for further general "Fourier basis"
// optimizations which should probably reside in QUnit, analogous to the |+>/|-> basis changes QUnit takes advantage of
// for "H" gates.

/** Caches controlled gate phase between shards, (as a case of "gate fusion" optimization particularly useful to QUnit)
 */
struct PhaseShard {
    real1 angle0;
    real1 angle1;
    bool isInvert;

    PhaseShard()
        : angle0(ZERO_R1)
        , angle1(ZERO_R1)
        , isInvert(false)
    {
    }
};

struct QEngineShard;
typedef QEngineShard* QEngineShardPtr;
typedef std::shared_ptr<PhaseShard> PhaseShardPtr;
typedef std::map<QEngineShardPtr, PhaseShardPtr> ShardToPhaseMap;

/** Associates a QInterface object with a set of bits. */
class QEngineShard : public ParallelFor {
public:
    QInterfacePtr unit;
    bitLenInt mapped;
    bool isEmulated;
    bool isProbDirty;
    bool isPhaseDirty;
    complex amp0;
    complex amp1;
    bool isPlusMinus;
    // Shards which this shard controls
    ShardToPhaseMap controlsShards;
    // Shards of which this shard is a target
    ShardToPhaseMap targetOfShards;

    QEngineShard()
        : unit(NULL)
        , mapped(0)
        , isEmulated(false)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , targetOfShards()
    {
    }

    QEngineShard(QInterfacePtr u, const bool& set)
        : unit(u)
        , mapped(0)
        , isEmulated(false)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , targetOfShards()
    {
        amp0 = set ? ZERO_CMPLX : ONE_CMPLX;
        amp1 = set ? ONE_CMPLX : ZERO_CMPLX;
    }

    // Dirty state constructor:
    QEngineShard(QInterfacePtr u, const bitLenInt& mapping)
        : unit(u)
        , mapped(mapping)
        , isEmulated(false)
        , isProbDirty(true)
        , isPhaseDirty(true)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , targetOfShards()
    {
    }

    void MakeDirty()
    {
        isProbDirty = true;
        isPhaseDirty = true;
    }

    bool ClampAmps(real1 norm_thresh)
    {
        bool didClamp = false;
        if (norm(amp0) < norm_thresh) {
            didClamp = true;
            amp0 = ZERO_R1;
            amp1 /= abs(amp1);
            if (!isProbDirty) {
                isPhaseDirty = false;
            }
        } else if (norm(amp1) < norm_thresh) {
            didClamp = true;
            amp1 = ZERO_R1;
            amp0 /= abs(amp0);
            if (!isProbDirty) {
                isPhaseDirty = false;
            }
        }
        return didClamp;
    }

    /// Remove another qubit as being a cached control of a phase gate buffer, for "this" as target bit.
    void RemovePhaseControl(QEngineShardPtr p)
    {
        ShardToPhaseMap::iterator phaseShard = targetOfShards.find(p);
        if (phaseShard != targetOfShards.end()) {
            phaseShard->first->controlsShards.erase(this);
            targetOfShards.erase(phaseShard);
        }
    }

    /// Remove another qubit as being a cached target of a phase gate buffer, for "this" as control bit.
    void RemovePhaseTarget(QEngineShardPtr p)
    {
        ShardToPhaseMap::iterator phaseShard = controlsShards.find(p);
        if (phaseShard != controlsShards.end()) {
            phaseShard->first->targetOfShards.erase(this);
            controlsShards.erase(phaseShard);
        }
    }

    /// Initialize a phase gate buffer, with "this" as target bit and a another qubit "p" as control
    void MakePhaseControlledBy(QEngineShardPtr p)
    {
        if (p && (targetOfShards.find(p) == targetOfShards.end())) {
            PhaseShardPtr ps = std::make_shared<PhaseShard>();
            targetOfShards[p] = ps;
            p->controlsShards[this] = ps;
        }
    }

    /// Initialize a phase gate buffer, with "this" as control bit and a another qubit "p" as target
    void MakePhaseControlOf(QEngineShardPtr p)
    {
        if (p && (controlsShards.find(p) == controlsShards.end())) {
            PhaseShardPtr ps = std::make_shared<PhaseShard>();
            controlsShards[p] = ps;
            p->targetOfShards[this] = ps;
        }
    }

    /// "Fuse" phase gate buffer angles, (and initialize the buffer, if necessary,) for the buffer with "this" as target
    /// bit and a another qubit as control
    void AddPhaseAngles(QEngineShardPtr control, real1 angle0Diff, real1 angle1Diff)
    {
        MakePhaseControlledBy(control);

        real1 nAngle0 = targetOfShards[control]->angle0 + angle0Diff;
        real1 nAngle1 = targetOfShards[control]->angle1 + angle1Diff;

        while (nAngle0 <= -M_PI) {
            nAngle0 += 2 * M_PI;
        }
        while (nAngle0 > M_PI) {
            nAngle0 -= 2 * M_PI;
        }
        while (nAngle1 <= -M_PI) {
            nAngle1 += 2 * M_PI;
        }
        while (nAngle1 > M_PI) {
            nAngle1 -= 2 * M_PI;
        }

        if ((nAngle0 == ZERO_R1) && (nAngle1 == ZERO_R1) && !targetOfShards[control]->isInvert) {
            // The buffer is equal to the identity operator, and it can be removed.
            RemovePhaseControl(control);
            return;
        }

        targetOfShards[control]->angle0 = nAngle0;
        targetOfShards[control]->angle1 = nAngle1;
    }

    void AddInversionAngles(QEngineShardPtr control, real1 angle0Diff, real1 angle1Diff)
    {
        MakePhaseControlledBy(control);

        PhaseShardPtr targetOfShard = targetOfShards[control];
        targetOfShard->isInvert = !targetOfShard->isInvert;
        std::swap(targetOfShard->angle0, targetOfShard->angle1);

        AddPhaseAngles(control, angle0Diff, angle1Diff);
    }

    bool isInvertControl()
    {
        ShardToPhaseMap::iterator phaseShard;
        for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                return true;
            }
        }

        return false;
    }

    bool isInvertTarget()
    {
        ShardToPhaseMap::iterator phaseShard;
        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                return true;
            }
        }

        return false;
    }

    bool isInvert() { return isInvertControl() || isInvertTarget(); }

    /// Take ambiguous control/target operations, and reintrepret them as targeting this bit
    void OptimizeControls()
    {
        QEngineShardPtr partner;
        real1 partnerAngle;

        ShardToPhaseMap tempControls = controlsShards;
        par_for(0, tempControls.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = tempControls.begin();
            std::advance(phaseShard, lcv);
            if ((isPlusMinus != phaseShard->first->isPlusMinus) || phaseShard->second->isInvert ||
                (phaseShard->second->angle0 != ZERO_R1)) {
                return;
            }

            partner = phaseShard->first;
            partnerAngle = phaseShard->second->angle1;

            phaseShard->first->targetOfShards.erase(this);
            controlsShards.erase(partner);

            AddPhaseAngles(partner, ZERO_R1, partnerAngle);
        });
    }

    /// If this bit is both control and target of another bit, try to combine the operations into one gate.
    void CombineGates()
    {
        ShardToPhaseMap::iterator partnerShard;
        QEngineShardPtr partner;
        real1 partnerAngle;

        ShardToPhaseMap tempControls = controlsShards;
        ShardToPhaseMap tempTargets = targetOfShards;
        par_for(0, tempControls.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = tempControls.begin();
            std::advance(phaseShard, lcv);

            if (isPlusMinus != phaseShard->first->isPlusMinus) {
                return;
            }

            partner = phaseShard->first;

            partnerShard = tempTargets.find(partner);
            if (partnerShard == tempTargets.end()) {
                return;
            }

            if (!phaseShard->second->isInvert && (phaseShard->second->angle0 == ZERO_R1)) {
                partnerAngle = phaseShard->second->angle1;

                phaseShard->first->targetOfShards.erase(this);
                controlsShards.erase(partner);

                AddPhaseAngles(partner, ZERO_R1, partnerAngle);
            } else if (!partnerShard->second->isInvert && (partnerShard->second->angle0 == ZERO_R1)) {
                partnerAngle = partnerShard->second->angle1;

                phaseShard->first->controlsShards.erase(this);
                targetOfShards.erase(partner);

                partner->AddPhaseAngles(this, ZERO_R1, partnerAngle);
            }
        });
    }

    /// If an "inversion" gate is applied to a qubit with controlled phase buffers, we can transform the buffers to
    /// commute, instead of incurring the cost of applying the buffers.
    void FlipPhaseAnti()
    {
        // These cases cannot be handled:
        // if (controlsShards.size() > 0) {
        //    return false;
        // }

        par_for(0, targetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = targetOfShards.begin();
            std::advance(phaseShard, lcv);
            std::swap(phaseShard->second->angle0, phaseShard->second->angle1);
        });
    }

    void CommutePhase(const complex& topLeft, const complex& bottomRight)
    {
        ShardToPhaseMap::iterator phaseShard;

        // These casess cannot be handled:
        // for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
        //    if (phaseShard->second->isInvert) {
        //        return false;
        //    }
        //}

        par_for(0, targetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = targetOfShards.begin();
            std::advance(phaseShard, lcv);
            if (!phaseShard->second->isInvert) {
                return;
            }

            phaseShard->second->angle0 =
                std::arg(std::polar(ONE_R1, phaseShard->second->angle0) * topLeft / bottomRight);
            phaseShard->second->angle1 =
                std::arg(std::polar(ONE_R1, phaseShard->second->angle1) * bottomRight / topLeft);
        });
    }

    // TODO: Just turn this into a QUnit method that flushes all the appropriate failed commutations, then remove them
    // from here.
    bool TryHCommute()
    {
        CombineGates();

        complex polar0, polar1;
        ShardToPhaseMap::iterator phaseShard;

        for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
            polar0 = std::polar(ONE_R1, phaseShard->second->angle0);
            polar1 = std::polar(ONE_R1, phaseShard->second->angle1);
            if (polar0 == polar1) {
                if (phaseShard->second->isInvert) {
                    return false;
                }
            } else if (polar0 == -polar1) {
                if (!phaseShard->second->isInvert) {
                    return false;
                }
            } else {
                return false;
            }
        }

        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            polar0 = std::polar(ONE_R1, phaseShard->second->angle0);
            polar1 = std::polar(ONE_R1, phaseShard->second->angle1);
            if ((polar0 != polar1) && (polar0 != -polar1)) {
                return false;
            }
        }

        bool didFlip;
        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            polar0 = std::polar(ONE_R1, phaseShard->second->angle0);
            polar1 = std::polar(ONE_R1, phaseShard->second->angle1);
            didFlip = false;
            if (polar0 == polar1) {
                if (phaseShard->second->isInvert) {
                    polar0 = (polar0 + polar1) / (real1)2;
                    polar1 = -polar0;
                    didFlip = true;
                }
            } else if (polar0 == -polar1) {
                if (!phaseShard->second->isInvert) {
                    polar0 = (polar0 + polar1) / (real1)2;
                    polar1 = polar0;
                    didFlip = true;
                }
            } else {
                return false;
            }

            if (didFlip) {
                phaseShard->second->isInvert = !phaseShard->second->isInvert;
                phaseShard->second->angle0 = (real1)arg(polar0);
                phaseShard->second->angle1 = (real1)arg(polar1);
            }
        }

        return true;
    }

    bool operator==(const QEngineShard& rhs) { return (mapped == rhs.mapped) && (unit == rhs.unit); }
    bool operator!=(const QEngineShard& rhs) { return (mapped != rhs.mapped) || (unit != rhs.unit); }
};

class QUnit;
typedef std::shared_ptr<QUnit> QUnitPtr;

class QUnit : public QInterface {
protected:
    QInterfaceEngine engine;
    QInterfaceEngine subengine;
    int devID;
    std::vector<QEngineShard> shards;
    complex phaseFactor;
    bool doNormalize;
    bool useHostRam;
    bool useRDRAND;
    bool isSparse;
    bool freezeBasis;

    virtual void SetQubitCount(bitLenInt qb)
    {
        shards.resize(qb);
        QInterface::SetQubitCount(qb);
    }

    QInterfacePtr MakeEngine(bitLenInt length, bitCapInt perm);

public:
    QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> ignored = {});
    QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> ignored = {});

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);
    using QInterface::Compose;
    virtual bitLenInt Compose(QUnitPtr toCopy, bool isConsumed = false);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bool isConsumed = false)
    {
        return Compose(std::dynamic_pointer_cast<QUnit>(toCopy), isConsumed);
    }
    virtual bitLenInt Compose(QUnitPtr toCopy, bitLenInt start, bool isConsumed = false);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start, bool isConsumed = false)
    {
        return Compose(std::dynamic_pointer_cast<QUnit>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decompose(start, length, std::dynamic_pointer_cast<QUnit>(dest));
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QUnitPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    using QInterface::H;
    virtual void H(bitLenInt target);
    using QInterface::X;
    virtual void X(bitLenInt target);
    using QInterface::Z;
    virtual void Z(bitLenInt target);
    using QInterface::CNOT;
    virtual void CNOT(bitLenInt control, bitLenInt target);
    using QInterface::AntiCNOT;
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);
    using QInterface::CCNOT;
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    using QInterface::AntiCCNOT;
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    using QInterface::CZ;
    virtual void CZ(bitLenInt control, bitLenInt target);

    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex);
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex);
    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);
    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);
    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubit);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    using QInterface::UniformlyControlledSingleBit;
    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    using QInterface::ForceM;
    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values, bool doApply = true);
    using QInterface::ForceMReg;
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true);

    /** @} */

    /**
     * \defgroup LogicGates Logic Gates
     *
     * Each bit is paired with a CL* variant that utilizes a classical bit as
     * an input.
     *
     * @{
     */

    using QInterface::AND;
    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length);
    using QInterface::OR;
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length);
    using QInterface::XOR;
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length);
    using QInterface::CLAND;
    virtual void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    using QInterface::CLOR;
    virtual void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    using QInterface::CLXOR;
    virtual void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
    virtual void PhaseFlip();
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values);
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1 Prob(bitLenInt qubit);
    virtual real1 ProbAll(bitCapInt fullRegister);
    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QUnit>(toCompare));
    }
    virtual bool ApproxCompare(QUnitPtr toCompare);
    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_threshold = REAL1_DEFAULT_ARG);
    virtual void Finish();
    virtual bool isFinished();

    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1);

    virtual QInterfacePtr Clone();

    /** @} */

protected:
    virtual void XBase(const bitLenInt& target);
    virtual void ZBase(const bitLenInt& target);
    virtual real1 ProbBase(const bitLenInt& qubit);

    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);

    typedef void (QInterface::*INCxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QInterface::*INCxxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QInterface::*CMULFn)(bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    typedef void (QInterface::*CMULModFn)(bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    void INT(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex, bool hasCarry,
        bitLenInt* controls = NULL, bitLenInt controlLen = 0);
    void INTS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex,
        bool hasCarry);
    void INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void INCxx(
        INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index);
    QInterfacePtr CMULEntangle(std::vector<bitLenInt> controlVec, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt>* controlsMapped);
    std::vector<bitLenInt> CMULEntangle(
        std::vector<bitLenInt> controlVec, bitLenInt start, bitCapInt carryStart, bitLenInt length);
    void CMULx(CMULFn fn, bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
        bitLenInt controlLen);
    void xMULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length, bool inverse);
    void CxMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen, bool inverse);
    void CMULModx(CMULModFn fn, bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt> controlVec);
    bool CArithmeticOptimize(bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>* controlVec);
    bool INTCOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex);
    bool INTSOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt overflowIndex);
    bool INTSCOptimize(
        bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex, bitLenInt overflowIndex);

    template <typename F>
    void CBoolReg(const bitLenInt& qInputStart, const bitCapInt& classicalInput, const bitLenInt& outputStart,
        const bitLenInt& length, F fn);

    virtual QInterfacePtr Entangle(std::vector<bitLenInt> bits);
    virtual QInterfacePtr Entangle(std::vector<bitLenInt*> bits);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2);
    virtual QInterfacePtr EntangleRange(
        bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3);
    virtual QInterfacePtr EntangleAll() { return EntangleRange(0, qubitCount); }

    virtual QInterfacePtr CloneBody(QUnitPtr copyPtr);

    virtual bool CheckBitPermutation(const bitLenInt& qubitIndex, const bool& inCurrentBasis = false);
    virtual bool CheckBitsPermutation(
        const bitLenInt& start, const bitLenInt& length, const bool& inCurrentBasis = false);
    virtual bitCapInt GetCachedPermutation(const bitLenInt& start, const bitLenInt& length);
    virtual bitCapInt GetCachedPermutation(const bitLenInt* bitArray, const bitLenInt& length);
    virtual bool CheckBitsPlus(const bitLenInt& qubitIndex, const bitLenInt& length);

    virtual QInterfacePtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    typedef bool (*ParallelUnitFn)(QInterfacePtr unit, real1 param1, real1 param2);
    bool ParallelUnitApply(ParallelUnitFn fn, real1 param1 = ZERO_R1, real1 param2 = ZERO_R1);

    virtual void SeparateBit(bool value, bitLenInt qubit, bool doDispose = true);

    void OrderContiguous(QInterfacePtr unit);

    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest);

    struct QSortEntry {
        bitLenInt bit;
        bitLenInt mapped;
        bool operator<(const QSortEntry& rhs) { return mapped < rhs.mapped; }
        bool operator>(const QSortEntry& rhs) { return mapped > rhs.mapped; }
    };
    void SortUnit(QInterfacePtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high);

    template <typename CF, typename F>
    void ApplyEitherControlled(const bitLenInt* controls, const bitLenInt& controlLen,
        const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F f, const bool& inCurrentBasis = false);

    bitCapInt GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    bitCapInt GetIndexedEigenstate(bitLenInt start, bitLenInt length, unsigned char* values);

    void Transform2x2(const complex* mtrxIn, complex* mtrxOut);
    void TransformPhase(const complex& topLeft, const complex& bottomRight, complex* mtrxOut);
    void TransformInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut);

    void TransformBasis1Qb(const bool& toPlusMinus, const bitLenInt& i);

    void RevertBasis2Qb(const bitLenInt& i, const bool& onlyInvert = false, const bool& onlyControlling = false,
        std::set<bitLenInt> exceptControlling = {}, std::set<bitLenInt> exceptTargetedBy = {},
        const bool& dumpSkipped = false);
    void ToPermBasis(const bitLenInt& i)
    {
        TransformBasis1Qb(false, i);
        RevertBasis2Qb(i);
    }
    void ToPermBasis(const bitLenInt& start, const bitLenInt& length)
    {
        bitLenInt i;
        for (i = 0; i < length; i++) {
            TransformBasis1Qb(false, start + i);
        }
        for (i = 0; i < length; i++) {
            RevertBasis2Qb(start + i);
        }
    }
    void ToPermBasisAll() { ToPermBasis(0, qubitCount); }
    void ToPermBasisMeasure(const bitLenInt& start, const bitLenInt& length)
    {
        if ((start == 0) && (length == qubitCount)) {
            ToPermBasisAllMeasure();
            return;
        }

        bitLenInt i;

        std::set<bitLenInt> exceptBits;
        for (i = 0; i < length; i++) {
            exceptBits.insert(start + i);
        }

        for (i = 0; i < length; i++) {
            TransformBasis1Qb(false, start + i);
        }
        for (i = 0; i < length; i++) {
            RevertBasis2Qb(start + i, true);
            RevertBasis2Qb(start + i, false, false, exceptBits, exceptBits, true);
        }
    }
    void ToPermBasisAllMeasure()
    {
        bitLenInt i;
        for (i = 0; i < qubitCount; i++) {
            TransformBasis1Qb(i, false);
        }
        for (i = 0; i < qubitCount; i++) {
            RevertBasis2Qb(i, true, false, {}, {}, true);
        }
    }
    void PopHBasis2Qb(const bitLenInt& i)
    {
        QEngineShard& shard = shards[i];
        if (shard.isPlusMinus && ((shard.targetOfShards.size() != 0) || (shard.controlsShards.size() != 0))) {
            TransformBasis1Qb(false, i);
        }
    }

    void CheckShardSeparable(const bitLenInt& target);

    void DirtyShardRange(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            shards[start + i].isProbDirty = true;
            shards[start + i].isPhaseDirty = true;
        }
    }

    void DirtyShardRangePhase(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            shards[start + i].isPhaseDirty = true;
        }
    }

    void DirtyShardIndexArray(bitLenInt* bitIndices, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            shards[bitIndices[i]].isProbDirty = true;
            shards[bitIndices[i]].isPhaseDirty = true;
        }
    }

    void DirtyShardIndexVector(std::vector<bitLenInt> bitIndices)
    {
        for (bitLenInt i = 0; i < bitIndices.size(); i++) {
            shards[bitIndices[i]].isProbDirty = true;
            shards[bitIndices[i]].isPhaseDirty = true;
        }
    }

    void EndEmulation(QEngineShard& shard)
    {
        if (shard.isEmulated) {
            complex bitState[2] = { shard.amp0, shard.amp1 };
            shard.unit->SetQuantumState(bitState);
            shard.isEmulated = false;
        }
    }

    void EndEmulation(const bitLenInt& target)
    {
        QEngineShard& shard = shards[target];
        EndEmulation(shard);
    }

    void EndEmulation(bitLenInt* bitIndices, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            EndEmulation(bitIndices[i]);
        }
    }

    void EndAllEmulation()
    {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            EndEmulation(i);
        }
    }

    template <typename F> void ApplyOrEmulate(QEngineShard& shard, F payload)
    {
        if ((shard.unit->GetQubitCount() == 1) && !shard.isProbDirty && !shard.isPhaseDirty) {
            shard.isEmulated = true;
        } else {
            payload(shard);
        }
    }

    bitLenInt FindShardIndex(const QEngineShard& shard)
    {
        for (bitLenInt i = 0; i < shards.size(); i++) {
            if (shards[i] == shard) {
                return i;
            }
        }
        return shards.size();
    }

    bool TryCnotOptimize(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
        const complex& topRight, const complex& bottomLeft, const bool& anti);

    void FlipPhaseAnti(const bitLenInt& target)
    {
        RevertBasis2Qb(target, false, true);
        shards[target].FlipPhaseAnti();
    }

    void CommutePhase(const bitLenInt& target, const complex& topLeft, const complex& bottomRight)
    {
        RevertBasis2Qb(target, true, true);
        shards[target].CommutePhase(topLeft, bottomRight);
    }

    /* Debugging and diagnostic routines. */
    void DumpShards();
    QInterfacePtr GetUnit(bitLenInt bit) { return shards[bit].unit; }
};

} // namespace Qrack
