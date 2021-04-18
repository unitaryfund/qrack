//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QEngineShard is the atomic qubit unit of the QUnit mapper. "PhaseShard" optimizations are basically just a very
// specific "gate fusion" type optimization, where multiple gates are composed into single product gates before
// application to the state vector, to reduce the total number of gates that need to be applied. Rather than handling
// this as a "QFusion" layer optimization, which will typically sit BETWEEN a base QEngine set of "shards" and a QUnit
// that owns them, this particular gate fusion optimization can be avoid representational entanglement in QUnit in the
// first place, which QFusion would not help with. Alternatively, another QFusion would have to be in place ABOVE the
// QUnit layer, (with QEngine "below,") for this to work. Additionally, QFusion is designed to handle more general gate
// fusion, not specifically controlled phase gates, which are entirely commuting among each other and possibly a
// jumping-off point for further general "Fourier basis" optimizations which should probably reside in QUnit, analogous
// to the |+>/|-> basis changes QUnit takes advantage of for "H" gates.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"

#define IS_SAME(c1, c2) (norm((c1) - (c2)) <= FP_NORM_EPSILON)
#define IS_OPPOSITE(c1, c2) (norm((c1) + (c2)) <= FP_NORM_EPSILON)
#define IS_ARG_0(c) IS_SAME(c, ONE_CMPLX)
#define IS_ARG_PI(c) IS_OPPOSITE(c, ONE_CMPLX)

namespace Qrack {

/** Caches controlled gate phase between shards, (as a case of "gate fusion" optimization particularly useful to QUnit)
 */
struct PhaseShard {
    complex cmplxDiff;
    complex cmplxSame;
    bool isInvert;

    PhaseShard()
        : cmplxDiff(ONE_CMPLX)
        , cmplxSame(ONE_CMPLX)
        , isInvert(false)
    {
    }
};

class QEngineShard;
typedef QEngineShard* QEngineShardPtr;
typedef std::shared_ptr<PhaseShard> PhaseShardPtr;
typedef std::map<QEngineShardPtr, PhaseShardPtr> ShardToPhaseMap;

/** Associates a QInterface object with a set of bits. */
class QEngineShard {
protected:
    typedef ShardToPhaseMap& (QEngineShard::*GetBufferFn)();
    typedef void (QEngineShard::*OptimizeFn)();
    typedef void (QEngineShard::*AddRemoveFn)(QEngineShardPtr);
    typedef void (QEngineShard::*AddAnglesFn)(
        QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame);

public:
    QInterfacePtr unit;
    bitLenInt mapped;
    bool isProbDirty;
    bool isPhaseDirty;
    complex amp0;
    complex amp1;
    bool isPauliX;
    bool isPauliY;
    // Shards which this shard controls
    ShardToPhaseMap controlsShards;
    // Shards which this shard (anti-)controls
    ShardToPhaseMap antiControlsShards;
    // Shards of which this shard is a target
    ShardToPhaseMap targetOfShards;
    // Shards of which this shard is an (anti-controlled) target
    ShardToPhaseMap antiTargetOfShards;
    // For FindShardIndex
    bool found;

protected:
    // We'd rather not have these getters at all, but we need their function pointers.
    ShardToPhaseMap& GetControlsShards() { return controlsShards; }
    ShardToPhaseMap& GetAntiControlsShards() { return antiControlsShards; }
    ShardToPhaseMap& GetTargetOfShards() { return targetOfShards; }
    ShardToPhaseMap& GetAntiTargetOfShards() { return antiTargetOfShards; }

public:
    QEngineShard()
        : unit(NULL)
        , mapped(0)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPauliX(false)
        , isPauliY(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
    }

    QEngineShard(const bool& set, const complex rand_phase = ONE_CMPLX)
        : unit(NULL)
        , mapped(0)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , isPauliX(false)
        , isPauliY(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
        amp0 = set ? ZERO_CMPLX : rand_phase;
        amp1 = set ? rand_phase : ZERO_CMPLX;
    }

    // Dirty state constructor:
    QEngineShard(QInterfacePtr u, const bitLenInt& mapping)
        : unit(u)
        , mapped(mapping)
        , isProbDirty(true)
        , isPhaseDirty(true)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPauliX(false)
        , isPauliY(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
    }

    void MakeDirty()
    {
        isProbDirty = true;
        isPhaseDirty = true;
    }

    bool ClampAmps(real1_f norm_thresh)
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

    void DumpSamePhaseBuffer(OptimizeFn optimizeFn, ShardToPhaseMap& localMap, AddRemoveFn remoteFn)
    {
        ((*this).*optimizeFn)();

        PhaseShardPtr buffer;
        ShardToPhaseMap::iterator phaseShard = localMap.begin();
        int lcv = 0;

        while (phaseShard != localMap.end()) {
            buffer = phaseShard->second;
            if (!buffer->isInvert && IS_SAME(buffer->cmplxDiff, buffer->cmplxSame)) {
                ((*this).*remoteFn)(phaseShard->first);
            } else {
                lcv++;
            }
            phaseShard = localMap.begin();
            std::advance(phaseShard, lcv);
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
    void DumpSamePhaseControlOf()
    {
        DumpSamePhaseBuffer(&QEngineShard::OptimizeTargets, controlsShards, &QEngineShard::RemovePhaseTarget);
    }
    void DumpSamePhaseAntiControlOf()
    {
        DumpSamePhaseBuffer(
            &QEngineShard::OptimizeAntiTargets, antiControlsShards, &QEngineShard::RemovePhaseAntiTarget);
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
    void AddAngles(QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame, AddRemoveFn localFn,
        ShardToPhaseMap& localMap, AddRemoveFn remoteFn)
    {
        ((*this).*localFn)(control);

        PhaseShardPtr targetOfShard = localMap[control];

        complex ncmplxDiff = targetOfShard->cmplxDiff * cmplxDiff;
        ncmplxDiff /= abs(ncmplxDiff);
        complex ncmplxSame = targetOfShard->cmplxSame * cmplxSame;
        ncmplxSame /= abs(ncmplxSame);

        if (!targetOfShard->isInvert && IS_ARG_0(ncmplxDiff) && IS_ARG_0(ncmplxSame)) {
            /* The buffer is equal to the identity operator, and it can be removed. */
            ((*this).*remoteFn)(control);
            return;
        }

        targetOfShard->cmplxDiff = ncmplxDiff;
        targetOfShard->cmplxSame = ncmplxSame;
    }

public:
    void AddPhaseAngles(QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame)
    {
        AddAngles(control, cmplxDiff, cmplxSame, &QEngineShard::MakePhaseControlledBy, targetOfShards,
            &QEngineShard::RemovePhaseControl);
    }
    void AddAntiPhaseAngles(QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame)
    {
        AddAngles(control, cmplxDiff, cmplxSame, &QEngineShard::MakePhaseAntiControlledBy, antiTargetOfShards,
            &QEngineShard::RemovePhaseAntiControl);
    }
    void AddInversionAngles(QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame)
    {
        MakePhaseControlledBy(control);
        targetOfShards[control]->isInvert = !targetOfShards[control]->isInvert;
        std::swap(targetOfShards[control]->cmplxDiff, targetOfShards[control]->cmplxSame);
        AddPhaseAngles(control, cmplxDiff, cmplxSame);
    }
    void AddAntiInversionAngles(QEngineShardPtr control, const complex& cmplxDiff, const complex& cmplxSame)
    {
        MakePhaseAntiControlledBy(control);
        antiTargetOfShards[control]->isInvert = !antiTargetOfShards[control]->isInvert;
        std::swap(antiTargetOfShards[control]->cmplxDiff, antiTargetOfShards[control]->cmplxSame);
        AddAntiPhaseAngles(control, cmplxDiff, cmplxSame);
    }

protected:
    void OptimizeBuffer(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet, AddAnglesFn phaseFn, bool makeThisControl)
    {
        PhaseShardPtr buffer;
        QEngineShardPtr partner;

        ShardToPhaseMap::iterator phaseShard;
        ShardToPhaseMap tempLocalMap = localMap;

        for (phaseShard = tempLocalMap.begin(); phaseShard != tempLocalMap.end(); phaseShard++) {
            buffer = phaseShard->second;
            partner = phaseShard->first;

            if (buffer->isInvert || !IS_ARG_0(buffer->cmplxDiff)) {
                continue;
            }

            ((*phaseShard->first).*remoteMapGet)().erase(this);
            localMap.erase(partner);

            if (makeThisControl) {
                ((*partner).*phaseFn)(this, ONE_CMPLX, buffer->cmplxSame);
            } else {
                ((*this).*phaseFn)(partner, ONE_CMPLX, buffer->cmplxSame);
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

    void OptimizeBothTargets()
    {
        PhaseShardPtr buffer;
        QEngineShardPtr partner;

        ShardToPhaseMap::iterator phaseShard;

        ShardToPhaseMap tempLocalMap = targetOfShards;
        for (phaseShard = tempLocalMap.begin(); phaseShard != tempLocalMap.end(); phaseShard++) {
            buffer = phaseShard->second;
            partner = phaseShard->first;

            if (buffer->isInvert) {
                continue;
            }

            if (IS_ARG_0(buffer->cmplxDiff)) {
                phaseShard->first->GetControlsShards().erase(this);
                targetOfShards.erase(partner);
                partner->AddPhaseAngles(this, ONE_CMPLX, buffer->cmplxSame);
            } else if (IS_ARG_0(buffer->cmplxSame)) {
                phaseShard->first->GetControlsShards().erase(this);
                targetOfShards.erase(partner);
                partner->AddAntiPhaseAngles(this, buffer->cmplxDiff, ONE_CMPLX);
            }
        }

        tempLocalMap = antiTargetOfShards;
        for (phaseShard = tempLocalMap.begin(); phaseShard != tempLocalMap.end(); phaseShard++) {
            buffer = phaseShard->second;
            partner = phaseShard->first;

            if (buffer->isInvert) {
                continue;
            }

            if (IS_ARG_0(buffer->cmplxDiff)) {
                phaseShard->first->GetAntiControlsShards().erase(this);
                antiTargetOfShards.erase(partner);
                partner->AddAntiPhaseAngles(this, ONE_CMPLX, buffer->cmplxSame);
            } else if (IS_ARG_0(buffer->cmplxSame)) {
                phaseShard->first->GetAntiControlsShards().erase(this);
                antiTargetOfShards.erase(partner);
                partner->AddPhaseAngles(this, buffer->cmplxDiff, ONE_CMPLX);
            }
        }
    }

protected:
    void CombineBuffers(GetBufferFn targetMapGet, GetBufferFn controlMapGet, AddAnglesFn angleFn)
    {
        PhaseShardPtr buffer1, buffer2;
        ShardToPhaseMap::iterator partnerShard;
        QEngineShardPtr partner;

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

            if (!buffer1->isInvert && IS_ARG_0(buffer1->cmplxDiff)) {
                ((*partner).*targetMapGet)().erase(this);
                ((*this).*controlMapGet)().erase(partner);
                ((*this).*angleFn)(partner, ONE_CMPLX, buffer1->cmplxSame);
            } else if (!buffer2->isInvert && IS_ARG_0(buffer2->cmplxDiff)) {
                ((*partner).*controlMapGet)().erase(this);
                ((*this).*targetMapGet)().erase(partner);
                ((*partner).*angleFn)(this, ONE_CMPLX, buffer2->cmplxSame);
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
            std::swap(phaseShard->second->cmplxDiff, phaseShard->second->cmplxSame);
            antiTargetOfShards[phaseShard->first] = phaseShard->second;
            targetOfShards.erase(phaseShard);
        } else if (phaseShard == targetOfShards.end()) {
            std::swap(antiPhaseShard->second->cmplxDiff, antiPhaseShard->second->cmplxSame);
            targetOfShards[antiPhaseShard->first] = antiPhaseShard->second;
            antiTargetOfShards.erase(antiPhaseShard);
        } else {
            std::swap(phaseShard->second->cmplxDiff, phaseShard->second->cmplxSame);
            std::swap(antiPhaseShard->second->cmplxDiff, antiPhaseShard->second->cmplxSame);
            std::swap(targetOfShards[control], antiTargetOfShards[control]);
        }
    }

    void FlipPhaseAnti()
    {
        std::unordered_set<QEngineShardPtr> toSwap;
        ShardToPhaseMap::iterator ctrlPhaseShard;
        for (ctrlPhaseShard = controlsShards.begin(); ctrlPhaseShard != controlsShards.end(); ctrlPhaseShard++) {
            toSwap.insert(ctrlPhaseShard->first);
        }
        for (ctrlPhaseShard = antiControlsShards.begin(); ctrlPhaseShard != antiControlsShards.end();
             ctrlPhaseShard++) {
            toSwap.insert(ctrlPhaseShard->first);
        }
        std::unordered_set<QEngineShardPtr>::iterator swapShard;
        for (swapShard = toSwap.begin(); swapShard != toSwap.end(); swapShard++) {
            (*swapShard)->SwapTargetAnti(this);
        }
        std::swap(controlsShards, antiControlsShards);

        ShardToPhaseMap::iterator phaseShard;

        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            std::swap(phaseShard->second->cmplxDiff, phaseShard->second->cmplxSame);
        }

        for (phaseShard = antiTargetOfShards.begin(); phaseShard != antiTargetOfShards.end(); phaseShard++) {
            std::swap(phaseShard->second->cmplxDiff, phaseShard->second->cmplxSame);
        }
    }

    void CommutePhase(const complex& topLeft, const complex& bottomRight)
    {
        ShardToPhaseMap::iterator phaseShard;
        PhaseShardPtr buffer;

        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            buffer = phaseShard->second;
            if (!buffer->isInvert) {
                return;
            }

            buffer->cmplxDiff *= topLeft / bottomRight;
            buffer->cmplxSame *= bottomRight / topLeft;
        }

        for (phaseShard = antiTargetOfShards.begin(); phaseShard != antiTargetOfShards.end(); phaseShard++) {
            buffer = phaseShard->second;
            if (!buffer->isInvert) {
                return;
            }

            buffer->cmplxDiff *= bottomRight / topLeft;
            buffer->cmplxSame *= topLeft / bottomRight;
        }
    }

protected:
    void RemoveIdentityBuffers(ShardToPhaseMap& localMap, GetBufferFn remoteMapGet)
    {
        PhaseShardPtr buffer;
        ShardToPhaseMap::iterator phaseShard = localMap.begin();

        while (phaseShard != localMap.end()) {
            buffer = phaseShard->second;
            if (!buffer->isInvert && IS_ARG_0(buffer->cmplxDiff) && IS_ARG_0(buffer->cmplxSame)) {
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
        ShardToPhaseMap::iterator phaseShard;
        PhaseShardPtr buffer;

        // See QUnit::CommuteH() for which cases cannot be commuted and are flushed.
        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            PhaseShardPtr buffer = phaseShard->second;
            if (norm(buffer->cmplxDiff - buffer->cmplxSame) < ONE_R1) {
                if (buffer->isInvert) {
                    buffer->cmplxSame = -buffer->cmplxDiff;
                    buffer->isInvert = false;
                }
            } else {
                if (buffer->isInvert) {
                    std::swap(buffer->cmplxDiff, buffer->cmplxSame);
                } else {
                    buffer->cmplxSame = buffer->cmplxDiff;
                    buffer->isInvert = true;
                }
            }
        }

        RemoveIdentityBuffers(targetOfShards, &QEngineShard::GetControlsShards);

        for (phaseShard = antiTargetOfShards.begin(); phaseShard != antiTargetOfShards.end(); phaseShard++) {
            PhaseShardPtr buffer = phaseShard->second;
            if (norm(buffer->cmplxDiff - buffer->cmplxSame) < ONE_R1) {
                if (buffer->isInvert) {
                    buffer->cmplxDiff = -buffer->cmplxSame;
                    buffer->isInvert = false;
                } else {
                }
            } else {
                if (buffer->isInvert) {
                    std::swap(buffer->cmplxDiff, buffer->cmplxSame);
                } else {
                    buffer->cmplxDiff = buffer->cmplxSame;
                    buffer->isInvert = true;
                }
            }
        }

        RemoveIdentityBuffers(antiTargetOfShards, &QEngineShard::GetAntiControlsShards);
    }

    bool IsInvertControlOf(QEngineShardPtr target) { return (controlsShards.find(target) != controlsShards.end()); }

    bool IsInvertAntiControlOf(QEngineShardPtr target)
    {
        return (antiControlsShards.find(target) != antiControlsShards.end());
    }

    bool IsInvertControl()
    {
        ShardToPhaseMap::iterator phaseShard;

        for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                return true;
            }
        }

        for (phaseShard = antiControlsShards.begin(); phaseShard != antiControlsShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                return true;
            }
        }

        return false;
    }

    bool IsInvertTarget()
    {
        ShardToPhaseMap::iterator phaseShard;

        for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                return true;
            }
        }

        for (phaseShard = antiTargetOfShards.begin(); phaseShard != antiTargetOfShards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                return true;
            }
        }

        return false;
    }

    bool IsInvert() { return IsInvertTarget() || IsInvertControl(); }

protected:
    void ClearMapInvertPhase(ShardToPhaseMap& shards)
    {
        ShardToPhaseMap::iterator phaseShard;

        for (phaseShard = shards.begin(); phaseShard != shards.end(); phaseShard++) {
            if (phaseShard->second->isInvert) {
                phaseShard->second->cmplxDiff = ONE_CMPLX;
                phaseShard->second->cmplxSame = ONE_CMPLX;
            }
        }
    }

public:
    void ClearInvertPhase()
    {
        // Upon measurement, buffered phase can sometimes be totally ignored.
        // If we clear phase before applying buffered inversions, we can optimize application as CNOT.
        ClearMapInvertPhase(controlsShards);
        ClearMapInvertPhase(antiControlsShards);
        ClearMapInvertPhase(targetOfShards);
        ClearMapInvertPhase(antiTargetOfShards);
    }

    bitLenInt GetQubitCount() { return unit ? unit->GetQubitCount() : 1U; };
    real1_f Prob()
    {
        if (!unit || !isProbDirty) {
            return norm(amp1);
        }

        return unit->Prob(mapped);
    }
    bool isClifford() { return unit ? unit->isClifford(mapped) : true; };
};

class QEngineShardMap {
protected:
    std::vector<QEngineShard> shards;
    std::vector<bitLenInt> swapMap;

public:
    QEngineShardMap()
    {
        // Intentionally left blank
    }

    QEngineShardMap(const bitLenInt& size)
        : shards(size)
        , swapMap(size)
    {
        for (bitLenInt i = 0; i < size; i++) {
            swapMap[i] = i;
        }
    }

    typedef std::vector<QEngineShard>::iterator iterator;

    QEngineShard& operator[](const bitLenInt& i) { return shards[swapMap[i]]; }

    iterator begin() { return shards.begin(); }

    iterator end() { return shards.end(); }

    bitLenInt size() { return shards.size(); }

    void push_back(const QEngineShard& shard)
    {
        shards.push_back(shard);
        swapMap.push_back(swapMap.size());
    }

    void insert(bitLenInt start, QEngineShardMap& toInsert)
    {
        bitLenInt oSize = size();

        shards.insert(shards.end(), toInsert.shards.begin(), toInsert.shards.end());
        swapMap.insert(swapMap.begin() + start, toInsert.swapMap.begin(), toInsert.swapMap.end());

        for (bitLenInt lcv = 0; lcv < toInsert.size(); lcv++) {
            swapMap[start + lcv] += oSize;
        }
    }

    void erase(bitLenInt begin, bitLenInt end)
    {
        bitLenInt offset, lcv;

        for (bitLenInt index = begin; index < end; index++) {
            offset = swapMap[index];
            shards.erase(shards.begin() + offset);

            for (lcv = 0; lcv < swapMap.size(); lcv++) {
                if (swapMap[lcv] >= offset) {
                    swapMap[lcv]--;
                }
            }
        }

        swapMap.erase(swapMap.begin() + begin, swapMap.begin() + end);
    }

    void swap(bitLenInt qubit1, bitLenInt qubit2) { std::swap(swapMap[qubit1], swapMap[qubit2]); }
};
} // namespace Qrack
