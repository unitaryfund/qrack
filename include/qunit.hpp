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
    complex cmplx0;
    complex cmplx1;
    bool isInvert;

    PhaseShard()
        : cmplx0(ONE_CMPLX)
        , cmplx1(ONE_CMPLX)
        , isInvert(false)
    {
    }
};

#define IS_ARG_0(c) (norm(c - ONE_CMPLX) <= amplitudeFloor)
#define IS_ARG_PI(c) (norm(c + ONE_CMPLX) <= amplitudeFloor)

class QEngineShard;
typedef QEngineShard* QEngineShardPtr;
typedef std::shared_ptr<PhaseShard> PhaseShardPtr;
typedef std::map<QEngineShardPtr, PhaseShardPtr> ShardToPhaseMap;

/** Associates a QInterface object with a set of bits. */
class QEngineShard : public ParallelFor {
protected:
    typedef ShardToPhaseMap& (QEngineShard::*GetBufferFn)();
    typedef void (QEngineShard::*OptimizeFn)();
    typedef void (QEngineShard::*AddRemoveFn)(QEngineShardPtr);
    typedef void (QEngineShard::*AddAnglesFn)(QEngineShardPtr control, const complex& cmplx0, const complex& cmplx1);

public:
    QInterfacePtr unit;
    bitLenInt mapped;
    real1 amplitudeFloor;
    bool isEmulated;
    bool isProbDirty;
    bool isPhaseDirty;
    complex amp0;
    complex amp1;
    bool isPlusMinus;
    // Shards which this shard controls
    ShardToPhaseMap controlsShards;
    // Shards which this shard (anti-)controls
    ShardToPhaseMap antiControlsShards;
    // Shards of which this shard is a target
    ShardToPhaseMap targetOfShards;
    // Shards of which this shard is an (anti-controlled) target
    ShardToPhaseMap antiTargetOfShards;

protected:
    // We'd rather not have these getters at all, but we need their function pointers.
    ShardToPhaseMap& GetControlsShards() { return controlsShards; }
    ShardToPhaseMap& GetAntiControlsShards() { return antiControlsShards; }
    ShardToPhaseMap& GetTargetOfShards() { return targetOfShards; }
    ShardToPhaseMap& GetAntiTargetOfShards() { return antiTargetOfShards; }

public:
    QEngineShard(const real1 amp_thresh = min_norm)
        : unit(NULL)
        , mapped(0)
        , amplitudeFloor(amp_thresh)
        , isEmulated(false)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
    {
    }

    QEngineShard(QInterfacePtr u, const bool& set, const real1 amp_thresh = min_norm)
        : unit(u)
        , mapped(0)
        , amplitudeFloor(amp_thresh)
        , isEmulated(false)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
    {
        amp0 = set ? ZERO_CMPLX : ONE_CMPLX;
        amp1 = set ? ONE_CMPLX : ZERO_CMPLX;
    }

    // Dirty state constructor:
    QEngineShard(QInterfacePtr u, const bitLenInt& mapping, const real1 amp_thresh = min_norm)
        : unit(u)
        , mapped(mapping)
        , amplitudeFloor(amp_thresh)
        , isEmulated(false)
        , isProbDirty(true)
        , isPhaseDirty(true)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
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

protected:
    void RemoveBuffer(QEngineShardPtr p, ShardToPhaseMap& localMap, GetBufferFn remoteMapGet)
    {
        ShardToPhaseMap::iterator phaseShard = localMap.find(p);
        if (phaseShard != localMap.end()) {
            ((*phaseShard->first).*remoteMapGet)().erase(this);
            localMap.erase(phaseShard);
        }
    }

public:
    void RemovePhaseControl(QEngineShardPtr p) { RemoveBuffer(p, targetOfShards, &QEngineShard::GetControlsShards); }
    void RemovePhaseTarget(QEngineShardPtr p) { RemoveBuffer(p, controlsShards, &QEngineShard::GetTargetOfShards); }
    void RemovePhaseAntiControl(QEngineShardPtr p)
    {
        RemoveBuffer(p, antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
    }
    void RemovePhaseAntiTarget(QEngineShardPtr p)
    {
        RemoveBuffer(p, antiControlsShards, &QEngineShard::GetAntiTargetOfShards);
    }

protected:
    void DumpBuffer(OptimizeFn optimizeFn, ShardToPhaseMap& localMap, AddRemoveFn remoteFn)
    {
        ((*this).*optimizeFn)();
        ShardToPhaseMap::iterator phaseShard = localMap.begin();
        while (phaseShard != localMap.end()) {
            ((*this).*remoteFn)(phaseShard->first);
            phaseShard = localMap.begin();
        }
    }

public:
    void DumpControlOf()
    {
        DumpBuffer(&QEngineShard::OptimizeTargets, controlsShards, &QEngineShard::RemovePhaseTarget);
    }
    void DumpAntiControlOf()
    {
        DumpBuffer(&QEngineShard::OptimizeAntiTargets, antiControlsShards, &QEngineShard::RemovePhaseAntiTarget);
    }
    void DumpTargetOf()
    {
        DumpBuffer(&QEngineShard::OptimizeControls, targetOfShards, &QEngineShard::RemovePhaseControl);
        DumpBuffer(&QEngineShard::OptimizeAntiControls, antiTargetOfShards, &QEngineShard::RemovePhaseAntiControl);
    }
    void DumpBuffers()
    {
        DumpControlOf();
        DumpAntiControlOf();
        DumpTargetOf();
    }

protected:
    void AddBuffer(QEngineShardPtr p, ShardToPhaseMap& localMap, GetBufferFn remoteFn)
    {
        if (p && (localMap.find(p) == localMap.end())) {
            PhaseShardPtr ps = std::make_shared<PhaseShard>();
            localMap[p] = ps;
            ((*p).*remoteFn)()[this] = ps;
        }
    }

public:
    void MakePhaseControlledBy(QEngineShardPtr p) { AddBuffer(p, targetOfShards, &QEngineShard::GetControlsShards); }
    void MakePhaseControlOf(QEngineShardPtr p) { AddBuffer(p, controlsShards, &QEngineShard::GetTargetOfShards); }
    void MakePhaseAntiControlledBy(QEngineShardPtr p)
    {
        AddBuffer(p, antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
    }
    void MakePhaseAntiControlOf(QEngineShardPtr p)
    {
        AddBuffer(p, antiControlsShards, &QEngineShard::GetAntiTargetOfShards);
    }

protected:
    void AddAngles(QEngineShardPtr control, const complex& cmplx0, const complex& cmplx1, AddRemoveFn localFn,
        ShardToPhaseMap& localMap, AddRemoveFn remoteFn)
    {
        ((*this).*localFn)(control);

        PhaseShardPtr targetOfShard = localMap[control];

        complex nCmplx0 = targetOfShard->cmplx0 * cmplx0;
        nCmplx0 /= abs(nCmplx0);
        complex nCmplx1 = targetOfShard->cmplx1 * cmplx1;
        nCmplx1 /= abs(nCmplx1);

        if (!targetOfShard->isInvert && IS_ARG_0(nCmplx0) &&
            IS_ARG_0(nCmplx1)) { /* The buffer is equal to the identity operator, and it can be removed. */
            ((*this).*remoteFn)(control);
            return;
        }

        targetOfShard->cmplx0 = nCmplx0;
        targetOfShard->cmplx1 = nCmplx1;
    }

public:
    void AddPhaseAngles(QEngineShardPtr control, const complex& cmplx0, const complex& cmplx1)
    {
        AddAngles(control, cmplx0, cmplx1, &QEngineShard::MakePhaseControlledBy, targetOfShards,
            &QEngineShard::RemovePhaseControl);
    }
    void AddAntiPhaseAngles(QEngineShardPtr control, const complex& cmplx0, const complex& cmplx1)
    {
        AddAngles(control, cmplx0, cmplx1, &QEngineShard::MakePhaseAntiControlledBy, antiTargetOfShards,
            &QEngineShard::RemovePhaseAntiControl);
    }

    void AddInversionAngles(QEngineShardPtr control, complex cmplx0, complex cmplx1)
    {
        MakePhaseControlledBy(control);

        PhaseShardPtr targetOfShard = targetOfShards[control];
        targetOfShard->isInvert = !targetOfShard->isInvert;
        std::swap(targetOfShard->cmplx0, targetOfShard->cmplx1);

        AddPhaseAngles(control, cmplx0, cmplx1);
    }

protected:
    void OptimizeBuffer(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet, AddAnglesFn phaseFn, bool makeThisControl)
    {
        PhaseShardPtr buffer;
        QEngineShardPtr partner;
        complex partnerAngle;

        ShardToPhaseMap::iterator phaseShard;
        ShardToPhaseMap tempLocalMap = localMap;

        for (phaseShard = tempLocalMap.begin(); phaseShard != tempLocalMap.end(); phaseShard++) {
            buffer = phaseShard->second;
            partner = phaseShard->first;

            if (buffer->isInvert || (isPlusMinus != partner->isPlusMinus) || !IS_ARG_0(buffer->cmplx0)) {
                continue;
            }

            partnerAngle = buffer->cmplx1;

            ((*phaseShard->first).*remoteMapGet)().erase(this);
            localMap.erase(partner);

            if (makeThisControl) {
                ((*partner).*phaseFn)(this, ONE_CMPLX, partnerAngle);
            } else {
                ((*this).*phaseFn)(partner, ONE_CMPLX, partnerAngle);
            }
        }
    }

public:
    void OptimizeControls()
    {
        OptimizeBuffer(controlsShards, &QEngineShard::GetTargetOfShards, &QEngineShard::AddPhaseAngles, false);
    }
    void OptimizeTargets()
    {
        OptimizeBuffer(targetOfShards, &QEngineShard::GetControlsShards, &QEngineShard::AddPhaseAngles, true);
    }
    void OptimizeAntiControls()
    {
        OptimizeBuffer(
            antiControlsShards, &QEngineShard::GetAntiTargetOfShards, &QEngineShard::AddAntiPhaseAngles, false);
    }
    void OptimizeAntiTargets()
    {
        OptimizeBuffer(
            antiTargetOfShards, &QEngineShard::GetAntiControlsShards, &QEngineShard::AddAntiPhaseAngles, true);
    }

protected:
    void CombineBuffers(GetBufferFn targetMapGet, GetBufferFn controlMapGet, AddAnglesFn angleFn)
    {
        PhaseShardPtr buffer1, buffer2;
        ShardToPhaseMap::iterator partnerShard;
        QEngineShardPtr partner;
        complex partnerAngle;

        ShardToPhaseMap::iterator phaseShard;
        ShardToPhaseMap tempControls = ((*this).*controlMapGet)();
        ShardToPhaseMap tempTargets = ((*this).*targetMapGet)();

        for (phaseShard = tempControls.begin(); phaseShard != tempControls.end(); phaseShard++) {
            partner = phaseShard->first;

            partnerShard = tempTargets.find(partner);
            if (partnerShard == tempTargets.end()) {
                continue;
            }

            buffer1 = phaseShard->second;
            buffer2 = partnerShard->second;

            if (!buffer1->isInvert && IS_ARG_0(buffer1->cmplx0)) {
                partnerAngle = buffer1->cmplx1;

                ((*partner).*targetMapGet)().erase(this);
                ((*this).*controlMapGet)().erase(partner);

                ((*this).*angleFn)(partner, ONE_CMPLX, partnerAngle);
            } else if (!buffer2->isInvert && IS_ARG_0(buffer2->cmplx0)) {
                partnerAngle = buffer2->cmplx1;

                ((*partner).*controlMapGet)().erase(this);
                ((*this).*targetMapGet)().erase(partner);

                ((*partner).*angleFn)(this, ONE_CMPLX, partnerAngle);
            }
        }
    }

public:
    /// If this bit is both control and target of another bit, try to combine the operations into one gate.
    void CombineGates()
    {
        CombineBuffers(
            &QEngineShard::GetTargetOfShards, &QEngineShard::GetControlsShards, &QEngineShard::AddPhaseAngles);
        CombineBuffers(&QEngineShard::GetAntiTargetOfShards, &QEngineShard::GetAntiControlsShards,
            &QEngineShard::AddAntiPhaseAngles);
    }

    void SwapTargetAnti(QEngineShardPtr control)
    {
        ShardToPhaseMap::iterator phaseShard = targetOfShards.find(control);
        ShardToPhaseMap::iterator antiPhaseShard = antiTargetOfShards.find(control);
        if (antiPhaseShard == antiTargetOfShards.end()) {
            antiTargetOfShards[phaseShard->first] = phaseShard->second;
            targetOfShards.erase(phaseShard);
        } else if (phaseShard == targetOfShards.end()) {
            targetOfShards[antiPhaseShard->first] = antiPhaseShard->second;
            antiTargetOfShards.erase(antiPhaseShard);
        } else {
            std::swap(targetOfShards[control], antiTargetOfShards[control]);
        }
    }

    void FlipPhaseAnti()
    {
        std::vector<QEngineShardPtr> alreadySwapped;
        ShardToPhaseMap::iterator phaseShard;
        for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
            phaseShard->first->SwapTargetAnti(this);
            alreadySwapped.push_back(phaseShard->first);
        }
        for (phaseShard = antiControlsShards.begin(); phaseShard != antiControlsShards.end(); phaseShard++) {
            if (std::find(alreadySwapped.begin(), alreadySwapped.end(), phaseShard->first) == alreadySwapped.end()) {
                phaseShard->first->SwapTargetAnti(this);
            }
        }
        std::swap(controlsShards, antiControlsShards);

        par_for(0, targetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = targetOfShards.begin();
            std::advance(phaseShard, lcv);
            std::swap(phaseShard->second->cmplx0, phaseShard->second->cmplx1);
        });

        par_for(0, antiTargetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = antiTargetOfShards.begin();
            std::advance(phaseShard, lcv);
            std::swap(phaseShard->second->cmplx0, phaseShard->second->cmplx1);
        });
    }

    void CommutePhase(const complex& topLeft, const complex& bottomRight)
    {
        par_for(0, targetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = targetOfShards.begin();
            std::advance(phaseShard, lcv);
            if (!phaseShard->second->isInvert) {
                return;
            }

            phaseShard->second->cmplx0 *= topLeft / bottomRight;
            phaseShard->second->cmplx1 *= bottomRight / topLeft;
        });

        par_for(0, antiTargetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = antiTargetOfShards.begin();
            std::advance(phaseShard, lcv);
            if (!phaseShard->second->isInvert) {
                return;
            }

            phaseShard->second->cmplx0 *= topLeft / bottomRight;
            phaseShard->second->cmplx1 *= bottomRight / topLeft;
        });
    }

protected:
    void RemoveIdentityBuffers(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet)
    {
        PhaseShardPtr buffer;
        ShardToPhaseMap::iterator phaseShard = localMap.begin();

        while (phaseShard != localMap.end()) {
            buffer = phaseShard->second;
            if (!buffer->isInvert && IS_ARG_0(buffer->cmplx0) && IS_ARG_0(buffer->cmplx1)) {
                // The buffer is equal to the identity operator, and it can be removed.
                ((*phaseShard->first).*remoteMapGet)().erase(this);
                localMap.erase(phaseShard);
            } else {
                phaseShard++;
            }
        }
    }

public:
    void CommuteH()
    {
        // See QUnit::CommuteH() for which cases cannot be commuted and are flushed.
        par_for(0, targetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = targetOfShards.begin();
            std::advance(phaseShard, lcv);
            PhaseShardPtr buffer = phaseShard->second;
            if (norm(buffer->cmplx0 - buffer->cmplx1) < ONE_R1) {
                if (buffer->isInvert) {
                    buffer->cmplx1 *= -ONE_CMPLX;
                    buffer->isInvert = false;
                }
            } else {
                if (buffer->isInvert) {
                    std::swap(buffer->cmplx0, buffer->cmplx1);
                } else {
                    buffer->cmplx1 = buffer->cmplx0;
                    buffer->isInvert = true;
                }
            }
        });

        RemoveIdentityBuffers(targetOfShards, &QEngineShard::GetControlsShards);

        par_for(0, antiTargetOfShards.size(), [&](const bitCapInt lcv, const int cpu) {
            ShardToPhaseMap::iterator phaseShard = antiTargetOfShards.begin();
            std::advance(phaseShard, lcv);
            PhaseShardPtr buffer = phaseShard->second;
            if (norm(buffer->cmplx0 - buffer->cmplx1) < ONE_R1) {
                if (buffer->isInvert) {
                    buffer->cmplx1 *= -ONE_CMPLX;
                    buffer->isInvert = false;
                }
            } else {
                if (buffer->isInvert) {
                    std::swap(buffer->cmplx0, buffer->cmplx1);
                } else {
                    buffer->cmplx1 = buffer->cmplx0;
                    buffer->isInvert = true;
                }
            }
        });

        RemoveIdentityBuffers(antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
    }

    bool IsInvertControlOf(QEngineShardPtr target)
    {
        bool toRet = false;

        ShardToPhaseMap::iterator phaseShard = controlsShards.find(target);
        if (phaseShard != controlsShards.end()) {
            toRet = phaseShard->second->isInvert;
        }

        if (toRet) {
            return true;
        }

        phaseShard = antiControlsShards.find(target);
        if (phaseShard != antiControlsShards.end()) {
            toRet |= phaseShard->second->isInvert;
        }

        return toRet;
    }

    bool IsInvertTargetOf(QEngineShardPtr control)
    {
        bool toRet = false;

        ShardToPhaseMap::iterator phaseShard = targetOfShards.find(control);
        if (phaseShard != targetOfShards.end()) {
            toRet = phaseShard->second->isInvert;
        }

        if (toRet) {
            return true;
        }

        phaseShard = antiTargetOfShards.find(control);
        if (phaseShard != antiTargetOfShards.end()) {
            toRet |= phaseShard->second->isInvert;
        }

        return toRet;
    }

    bool IsInvertControl()
    {
        bool toRet = false;
        ShardToPhaseMap::iterator phaseShard;

        for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                toRet = true;
                break;
            }
        }

        if (toRet) {
            return true;
        }

        for (phaseShard = antiControlsShards.begin(); phaseShard != antiControlsShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                toRet = true;
                break;
            }
        }

        return toRet;
    }

    bool IsInvertTarget()
    {
        bool toRet = false;
        ShardToPhaseMap::iterator phaseShard;

        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                toRet = true;
                break;
            }
        }

        if (toRet) {
            return true;
        }

        for (phaseShard = antiTargetOfShards.begin(); phaseShard != antiTargetOfShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                toRet = true;
                break;
            }
        }

        return false;
    }

    bool IsInvert() { return IsInvertControl() || IsInvertTarget(); }

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

    virtual ~QUnit() { Dump(); }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);
    using QInterface::Compose;
    virtual bitLenInt Compose(QUnitPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QUnit>(toCopy)); }
    virtual bitLenInt Compose(QUnitPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
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
    using QInterface::CCZ;
    virtual void CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target);

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
    virtual void Dump();

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

    enum RevertExclusivity { INVERT_AND_PHASE = 0, ONLY_INVERT = 1, ONLY_PHASE = 2 };
    enum RevertControl { CONTROLS_AND_TARGETS = 0, ONLY_CONTROLS = 1, ONLY_TARGETS = 2 };
    enum RevertAnti { CTRL_AND_ANTI = 0, ONLY_CTRL = 1, ONLY_ANTI = 2 };

    void ApplyBuffer(
        ShardToPhaseMap::iterator phaseShard, const bitLenInt& control, const bitLenInt& target, const bool& isAnti);
    void ApplyBufferMap(const bitLenInt& bitIndex, ShardToPhaseMap bufferMap, const RevertExclusivity& exclusivity,
        const bool& isControl, const bool& isAnti, std::set<bitLenInt> exceptPartners, const bool& dumpSkipped);
    void RevertBasis2Qb(const bitLenInt& i, const RevertExclusivity& exclusivity = INVERT_AND_PHASE,
        const RevertControl& controlExclusivity = CONTROLS_AND_TARGETS,
        const RevertAnti& antiExclusivity = CTRL_AND_ANTI, std::set<bitLenInt> exceptControlling = {},
        std::set<bitLenInt> exceptTargetedBy = {}, const bool& dumpSkipped = false);

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
            RevertBasis2Qb(start + i, ONLY_INVERT);
            RevertBasis2Qb(
                start + i, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, exceptBits, exceptBits, true);
        }
    }
    void ToPermBasisAllMeasure()
    {
        bitLenInt i;
        for (i = 0; i < qubitCount; i++) {
            TransformBasis1Qb(i, false);
        }
        for (i = 0; i < qubitCount; i++) {
            RevertBasis2Qb(i, ONLY_INVERT, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, {}, true);
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

    void CommuteH(const bitLenInt& bitIndex);

    /* Debugging and diagnostic routines. */
    void DumpShards();
    QInterfacePtr GetUnit(bitLenInt bit) { return shards[bit].unit; }
};

} // namespace Qrack
