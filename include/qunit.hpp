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
#include <unordered_set>

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

#define IS_SAME(c1, c2) (norm((c1) - (c2)) <= amplitudeThreshold)
#define IS_OPPOSITE(c1, c2) (norm((c1) + (c2)) <= amplitudeThreshold)
#define IS_ARG_0(c) IS_SAME(c, ONE_CMPLX)
#define IS_ARG_PI(c) IS_OPPOSITE(c, ONE_CMPLX)

class QEngineShard;
typedef QEngineShard* QEngineShardPtr;
typedef std::shared_ptr<PhaseShard> PhaseShardPtr;
typedef std::map<QEngineShardPtr, PhaseShardPtr> ShardToPhaseMap;
typedef std::shared_ptr<bool> BoolPtr;

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
    real1 amplitudeThreshold;
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
    BoolPtr isClifford;
    // For FindShardIndex
    bool found;

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
        , amplitudeThreshold(amp_thresh)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
        isClifford = std::make_shared<bool>(true);
    }

    QEngineShard(const bool& set, const real1 amp_thresh = min_norm)
        : unit(NULL)
        , mapped(0)
        , amplitudeThreshold(amp_thresh)
        , isProbDirty(false)
        , isPhaseDirty(false)
        , isPlusMinus(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , found(false)
    {
        isClifford = std::make_shared<bool>(true);
        amp0 = set ? ZERO_CMPLX : ONE_CMPLX;
        amp1 = set ? ONE_CMPLX : ZERO_CMPLX;
    }

    // Dirty state constructor:
    QEngineShard(QInterfacePtr u, const bitLenInt& mapping, BoolPtr isClif, const real1 amp_thresh = min_norm)
        : unit(u)
        , mapped(mapping)
        , amplitudeThreshold(amp_thresh)
        , isProbDirty(true)
        , isPhaseDirty(true)
        , amp0(ONE_CMPLX)
        , amp1(ZERO_CMPLX)
        , isPlusMinus(false)
        , controlsShards()
        , antiControlsShards()
        , targetOfShards()
        , antiTargetOfShards()
        , isClifford(isClif)
        , found(false)
    {
    }

    void MakeDirty()
    {
        if (isProbDirty || isPhaseDirty || !(*isClifford)) {
            isProbDirty = true;
        } else {
            amp0 = M_SQRT1_2;
            amp1 = M_SQRT1_2;
        }
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

    bitLenInt GetQubitCount() { return unit ? unit->GetQubitCount() : 1U; };
    real1 Prob()
    {
        if (!isProbDirty) {
            return norm(amp1);
        }

        return unit->Prob(mapped);
    };
};

class QUnit;
typedef std::shared_ptr<QUnit> QUnitPtr;

class QUnit : public QInterface {
protected:
    QInterfaceEngine engine;
    QInterfaceEngine subEngine;
    int devID;
    std::vector<QEngineShard> shards;
    complex phaseFactor;
    bool doNormalize;
    bool useHostRam;
    bool useRDRAND;
    bool isSparse;
    bool freezeBasisH;
    bool freezeBasis2Qb;
    bitLenInt thresholdQubits;

    QInterfacePtr MakeEngine(bitLenInt length, bitCapInt perm);

public:
    QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0);
    QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0)
        : QUnit(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold)
    {
    }

    virtual ~QUnit() { Dump(); }

    virtual void SetConcurrency(uint32_t threadsPerEngine)
    {
        ParallelUnitApply(
            [](QInterfacePtr unit, real1 unused1, real1 unused2, int32_t threadsPerEngine) {
                unit->SetConcurrency(threadsPerEngine);
                return true;
            },
            ZERO_R1, ZERO_R1, threadsPerEngine);
    }

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
    using QInterface::CH;
    virtual void CH(bitLenInt control, bitLenInt target);

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
    virtual void FSim(real1 theta, real1 phi, bitLenInt qubit1, bitLenInt qubit2);

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

    typedef bool (*ParallelUnitFn)(QInterfacePtr unit, real1 param1, real1 param2, int32_t param3);
    bool ParallelUnitApply(ParallelUnitFn fn, real1 param1 = ZERO_R1, real1 param2 = ZERO_R1, int32_t param3 = 0);

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
        const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F f, const bool& isClifford,
        const bool& inCurrentBasis = false);

    bitCapInt GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    bitCapInt GetIndexedEigenstate(bitLenInt start, bitLenInt length, unsigned char* values);

    void Transform2x2(const complex* mtrxIn, complex* mtrxOut);
    void TransformPhase(const complex& topLeft, const complex& bottomRight, complex* mtrxOut);
    void TransformInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut);

    void RevertBasis1Qb(const bitLenInt& i)
    {
        if (freezeBasisH || !shards[i].isPlusMinus) {
            // Recursive call that should be blocked,
            // or already in target basis.
            return;
        }

        freezeBasisH = true;
        H(i);
        shards[i].isPlusMinus = false;
        freezeBasisH = false;
    }

    enum RevertExclusivity { INVERT_AND_PHASE = 0, ONLY_INVERT = 1, ONLY_PHASE = 2 };
    enum RevertControl { CONTROLS_AND_TARGETS = 0, ONLY_CONTROLS = 1, ONLY_TARGETS = 2 };
    enum RevertAnti { CTRL_AND_ANTI = 0, ONLY_CTRL = 1, ONLY_ANTI = 2 };

    void ApplyBuffer(PhaseShardPtr phaseShard, const bitLenInt& control, const bitLenInt& target, const bool& isAnti);
    void ApplyBufferMap(const bitLenInt& bitIndex, ShardToPhaseMap bufferMap, const RevertExclusivity& exclusivity,
        const bool& isControl, const bool& isAnti, std::set<bitLenInt> exceptPartners, const bool& dumpSkipped);
    void RevertBasis2Qb(const bitLenInt& i, const RevertExclusivity& exclusivity = INVERT_AND_PHASE,
        const RevertControl& controlExclusivity = CONTROLS_AND_TARGETS,
        const RevertAnti& antiExclusivity = CTRL_AND_ANTI, std::set<bitLenInt> exceptControlling = {},
        std::set<bitLenInt> exceptTargetedBy = {}, const bool& dumpSkipped = false, const bool& skipOptimized = false);

    void Flush0Eigenstate(const bitLenInt& i)
    {
        shards[i].DumpControlOf();
        if (randGlobalPhase) {
            shards[i].DumpSamePhaseAntiControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_ANTI);
    }
    void Flush1Eigenstate(const bitLenInt& i)
    {
        shards[i].DumpAntiControlOf();
        if (randGlobalPhase) {
            shards[i].DumpSamePhaseControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_CTRL);
    }
    void ToPermBasis(const bitLenInt& i)
    {
        RevertBasis1Qb(i);
        RevertBasis2Qb(i);
    }
    void ToPermBasis(const bitLenInt& start, const bitLenInt& length)
    {
        bitLenInt i;
        for (i = 0; i < length; i++) {
            RevertBasis1Qb(start + i);
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
            RevertBasis1Qb(start + i);
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
            RevertBasis1Qb(i);
        }
        for (i = 0; i < qubitCount; i++) {
            RevertBasis2Qb(i, ONLY_INVERT, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, {}, true);
        }
    }

    void DirtyShardRange(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            shards[start + i].MakeDirty();
        }
    }

    void DirtyShardRangePhase(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            shards[start + i].isPhaseDirty = true;
        }
    }

    void DirtyShardIndexVector(std::vector<bitLenInt> bitIndices)
    {
        for (bitLenInt i = 0; i < bitIndices.size(); i++) {
            shards[bitIndices[i]].MakeDirty();
        }
    }

    void EndEmulation(QEngineShard& shard)
    {
        if (!shard.unit) {
            complex bitState[2] = { shard.amp0, shard.amp1 };
            shard.unit = MakeEngine(1, 0);
            shard.unit->SetQuantumState(bitState);
        }
    }

    void EndEmulation(const bitLenInt& target)
    {
        QEngineShard& shard = shards[target];
        EndEmulation(shard);
    }

    void EndEmulation(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            EndEmulation(start + i);
        }
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

    bitLenInt FindShardIndex(QEngineShardPtr shard)
    {
        shard->found = true;
        for (bitLenInt i = 0; i < shards.size(); i++) {
            if (shards[i].found) {
                shard->found = false;
                return i;
            }
        }
        shard->found = false;
        return shards.size();
    }

    void CommuteH(const bitLenInt& bitIndex);
};

} // namespace Qrack
