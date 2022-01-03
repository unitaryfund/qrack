//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// When we allocate a quantum register, all bits are in a (re)set state. At this point,
// we know they are separable, in the sense of full Schmidt decomposability into qubits
// in the "natural" or "permutation" basis of the register. Many operations can be
// traced in terms of fewer qubits that the full "Schr\{"o}dinger representation."
//
// Based on experimentation, QUnit is designed to avoid increasing representational
// entanglement for its primary action, and only try to decrease it when inquiries
// about probability need to be made otherwise anyway. Avoiding introducing the cost of
// basically any entanglement whatsoever, rather than exponentially costly "garbage
// collection," should be the first and ultimate concern, in the authors' experience.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#include <ctime>
#include <initializer_list>
#include <map>

#define DIRTY(shard) (shard.isPhaseDirty || shard.isProbDirty)
#define IS_AMP_0(c) (norm(c) <= separabilityThreshold)
#define IS_0_R1(r) (r == ZERO_R1)
#define IS_1_R1(r) (r == ONE_R1)
#define IS_1_CMPLX(c) (norm((c)-ONE_CMPLX) <= FP_NORM_EPSILON)
#define SHARD_STATE(shard) (norm(shard.amp0) < (ONE_R1 / 2))
#define QUEUED_PHASE(shard)                                                                                            \
    ((shard.targetOfShards.size() != 0) || (shard.controlsShards.size() != 0) ||                                       \
        (shard.antiTargetOfShards.size() != 0) || (shard.antiControlsShards.size() != 0))
#define CACHED_Z(shard) (!shard.isPauliX && !shard.isPauliY && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_ZERO_OR_ONE(shard) (CACHED_Z(shard) && (IS_AMP_0(shard.amp0) || IS_AMP_0(shard.amp1)))
#define CACHED_ZERO(shard) (CACHED_Z(shard) && IS_AMP_0(shard.amp1))
#define CACHED_ONE(shard) (CACHED_Z(shard) && IS_AMP_0(shard.amp0))
#define CACHED_X(shard) (shard.isPauliX && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_PLUS_OR_MINUS(shard) (CACHED_X(shard) && (IS_AMP_0(shard.amp0) || IS_AMP_0(shard.amp1)))
#define CACHED_PLUS(shard) (CACHED_X(shard) && IS_AMP_0(shard.amp1))
/* "UNSAFE" variants here do not check whether the bit is in |0>/|1> rather than |+>/|-> basis. */
#define UNSAFE_CACHED_ZERO_OR_ONE(shard)                                                                               \
    (!shard.isProbDirty && !shard.isPauliX && !shard.isPauliY && (IS_AMP_0(shard.amp0) || IS_AMP_0(shard.amp1)))
#define UNSAFE_CACHED_X(shard)                                                                                         \
    (!shard.isProbDirty && shard.isPauliX && !shard.isPauliY && (IS_AMP_0(shard.amp0) || IS_AMP_0(shard.amp1)))
#define UNSAFE_CACHED_ONE(shard) (!shard.isProbDirty && !shard.isPauliX && !shard.isPauliY && IS_AMP_0(shard.amp0))
#define UNSAFE_CACHED_ZERO(shard) (!shard.isProbDirty && !shard.isPauliX && !shard.isPauliY && IS_AMP_0(shard.amp1))
#define IS_SAME_UNIT(shard1, shard2) (shard1.unit && (shard1.unit == shard2.unit))
#define ARE_CLIFFORD(shard1, shard2)                                                                                   \
    ((engines[0] == QINTERFACE_STABILIZER_HYBRID) && (shard1.isClifford() || shard2.isClifford()))
#define BLOCKED_SEPARATE(shard) (shard.unit && shard.unit->isClifford() && !shard.unit->TrySeparate(shard.mapped))

namespace Qrack {

QUnit::QUnit(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , engines(eng)
    , unpagedEngines()
    , devID(deviceID)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , freezeBasis2Qb(false)
    , freezeTrySeparate(false)
    , isReactiveSeparate(true)
    , thresholdQubits(qubitThreshold)
    , separabilityThreshold(sep_thresh)
    , deviceIDs(devList)
{
#if ENABLE_ENV_VARS
    if (getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")) {
        separabilityThreshold = (real1_f)std::stof(std::string(getenv("QRACK_QUNIT_SEPARABILITY_THRESHOLD")));
    }
#endif

#if ENABLE_OPENCL
    if ((engines.size() == 1U) && (engines[0] == QINTERFACE_STABILIZER_HYBRID)) {
        bitLenInt segmentGlobalQb = 0U;
#if ENABLE_ENV_VARS
        if (getenv("QRACK_SEGMENT_GLOBAL_QB")) {
            segmentGlobalQb = (bitLenInt)std::stoi(std::string(getenv("QRACK_SEGMENT_GLOBAL_QB")));
        }
#endif

        DeviceContextPtr devContext = OCLEngine::Instance().GetDeviceContextPtr(devID);
        bitLenInt maxPageQubits = log2(devContext->GetMaxAlloc() / sizeof(complex)) - segmentGlobalQb;
        if (qubitCount > maxPageQubits) {
            engines.push_back(QINTERFACE_QPAGER);
        }
    }
#endif

    SetPermutation(initState);
}

QInterfacePtr QUnit::MakeEngine(bitLenInt length, bitCapInt perm)
{
    return CreateQuantumInterface(engines, length, perm, rand_generator, phaseFactor, doNormalize, randGlobalPhase,
        useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    Dump();

    shards = QEngineShardMap();

    for (bitLenInt i = 0; i < qubitCount; i++) {
        bool bitState = ((perm >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards.push_back(QEngineShard(bitState, GetNonunitaryPhase()));
    }
}

void QUnit::SetQuantumState(const complex* inputState)
{
    Dump();

    if (qubitCount == 1U) {
        QEngineShard& shard = shards[0];
        shard.unit = NULL;
        shard.mapped = 0;
        shard.isProbDirty = false;
        shard.isPhaseDirty = false;
        shard.amp0 = inputState[0];
        shard.amp1 = inputState[1];
        shard.isPauliX = false;
        shard.isPauliY = false;
        if (IS_AMP_0(shard.amp0 - shard.amp1)) {
            shard.isPauliX = true;
            shard.isPauliY = false;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_AMP_0(shard.amp0 + shard.amp1)) {
            shard.isPauliX = true;
            shard.isPauliY = false;
            shard.amp1 = shard.amp0 / abs(shard.amp0);
            shard.amp0 = ZERO_R1;
        } else if (IS_AMP_0((I_CMPLX * inputState[0]) - inputState[1])) {
            shard.isPauliX = false;
            shard.isPauliY = true;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_AMP_0((I_CMPLX * inputState[0]) + inputState[1])) {
            shard.isPauliX = false;
            shard.isPauliY = true;
            shard.amp1 = shard.amp0 / abs(shard.amp0);
            shard.amp0 = ZERO_R1;
        }
        return;
    }

    QInterfacePtr unit = MakeEngine(qubitCount, 0);
    unit->SetQuantumState(inputState);

    for (bitLenInt idx = 0; idx < qubitCount; idx++) {
        shards[idx] = QEngineShard(unit, idx);
    }
}

void QUnit::GetQuantumState(complex* outputState)
{
    if (qubitCount == 1U) {
        RevertBasis1Qb(0);
        if (!shards[0].unit) {
            outputState[0] = shards[0].amp0;
            outputState[1] = shards[0].amp1;

            return;
        }
    }

    QUnitPtr thisCopyShared;
    QUnit* thisCopy;

    if (shards[0].GetQubitCount() == qubitCount) {
        ToPermBasisAll();
        OrderContiguous(shards[0].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnit>(Clone());
        thisCopyShared->EntangleAll();
        thisCopy = thisCopyShared.get();
    }

    thisCopy->shards[0].unit->GetQuantumState(outputState);
}

void QUnit::GetProbs(real1* outputProbs)
{
    if (qubitCount == 1U) {
        RevertBasis1Qb(0);
        if (!shards[0].unit) {
            outputProbs[0] = norm(shards[0].amp0);
            outputProbs[1] = norm(shards[0].amp1);

            return;
        }
    }

    QUnitPtr thisCopyShared;
    QUnit* thisCopy;

    if (shards[0].GetQubitCount() == qubitCount) {
        ToPermBasisProb();
        OrderContiguous(shards[0].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnit>(Clone());
        thisCopyShared->EntangleAll(true);
        thisCopy = thisCopyShared.get();
    }

    thisCopy->shards[0].unit->GetProbs(outputProbs);
}

complex QUnit::GetAmplitude(bitCapInt perm) { return GetAmplitudeOrProb(perm, false); }

complex QUnit::GetAmplitudeOrProb(bitCapInt perm, bool isProb)
{
    if (isProb) {
        ToPermBasisProb();
    } else {
        ToPermBasisAll();
    }

    complex result(ONE_R1, ZERO_R1);

    std::map<QInterfacePtr, bitCapInt> perms;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        QEngineShard& shard = shards[i];

        if (!shard.unit) {
            result *= ((perm >> (bitCapIntOcl)i) & ONE_BCI) ? shard.amp1 : shard.amp0;
            continue;
        }

        if (perms.find(shard.unit) == perms.end()) {
            perms[shard.unit] = 0U;
        }
        if ((perm >> (bitCapIntOcl)i) & ONE_BCI) {
            perms[shard.unit] |= pow2(shard.mapped);
        }
    }

    for (auto&& qi : perms) {
        result *= qi.first->GetAmplitude(qi.second);
        if (IS_AMP_0(result)) {
            break;
        }
    }

    if ((shards[0].GetQubitCount() > 1) && IS_1_R1(norm(result)) && (randGlobalPhase || IS_AMP_0(result - ONE_CMPLX))) {
        SetPermutation(perm);
    }

    return result;
}

void QUnit::SetAmplitude(bitCapInt perm, complex amp)
{
    EntangleAll();
    shards[0].unit->SetAmplitude(perm, amp);
}

bitLenInt QUnit::Compose(QUnitPtr toCopy) { return Compose(toCopy, qubitCount); }

/*
 * Append QInterface in the middle of QUnit.
 */
bitLenInt QUnit::Compose(QUnitPtr toCopy, bitLenInt start)
{
    /* Create a clone of the quantum state in toCopy. */
    QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(toCopy->Clone());

    /* Insert the new shards in the middle */
    shards.insert(start, clone->shards);

    SetQubitCount(qubitCount + toCopy->GetQubitCount());

    return start;
}

void QUnit::Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
{
    for (bitLenInt i = 0; i < length; i++) {
        RevertBasis2Qb(start + i);
    }

    // Move "emulated" bits immediately into the destination, which is initialized.
    // Find a set of shard "units" to order contiguously. Also count how many bits to decompose are in each subunit.
    std::map<QInterfacePtr, bitLenInt> subunits;
    for (bitLenInt i = 0; i < length; i++) {
        QEngineShard& shard = shards[start + i];
        if (shard.unit) {
            subunits[shard.unit]++;
        } else if (dest) {
            dest->shards[i] = shard;
        }
    }

    // Order the subsystem units contiguously. (They might be entangled at random with bits not involed in the
    // operation.)
    for (auto subunit = subunits.begin(); subunit != subunits.end(); subunit++) {
        OrderContiguous(subunit->first);
    }

    // After ordering all subunits contiguously, since the top level mapping is a contiguous array, all subunit sets are
    // also contiguous. From the lowest index bits, they are mapped simply for the length count of bits involved in the
    // entire subunit.
    std::map<QInterfacePtr, bitLenInt> decomposedUnits;
    for (bitLenInt i = 0; i < length; i++) {
        QEngineShard& shard = shards[start + i];
        QInterfacePtr unit = shard.unit;

        if (unit == NULL) {
            continue;
        }

        if (decomposedUnits.find(unit) == decomposedUnits.end()) {
            decomposedUnits[unit] = start + i;
            bitLenInt subLen = subunits[unit];
            bitLenInt origLen = unit->GetQubitCount();
            if (subLen != origLen) {
                if (dest) {
                    QInterfacePtr nUnit = MakeEngine(subLen, 0);
                    shard.unit->Decompose(shard.mapped, nUnit);
                    shard.unit = nUnit;
                } else {
                    shard.unit->Dispose(shard.mapped, subLen);
                }

                if ((subLen == 1U) && dest) {
                    complex amps[2];
                    shard.unit->GetQuantumState(amps);
                    shard.amp0 = amps[0];
                    shard.amp1 = amps[1];
                    shard.isProbDirty = false;
                    shard.isPhaseDirty = false;
                    shard.unit = NULL;
                    shard.mapped = 0;
                    if (doNormalize) {
                        shard.ClampAmps(amplitudeFloor);
                    }
                }

                if (subLen == (origLen - 1U)) {
                    QEngineShard* pShard = NULL;
                    bitLenInt mapped = shards[decomposedUnits[unit]].mapped;
                    if (mapped == 0) {
                        mapped += subLen;
                    } else {
                        mapped = 0;
                    }
                    for (bitLenInt i = 0; i < shards.size(); i++) {
                        if ((shards[i].unit == unit) && (shards[i].mapped == mapped)) {
                            pShard = &shards[i];
                            break;
                        }
                    }

                    if (!pShard) {
                        // This should never happen.
                        throw std::runtime_error("QUnit::Detach found NULL pShard.");
                    }

                    complex amps[2];
                    pShard->unit->GetQuantumState(amps);
                    pShard->amp0 = amps[0];
                    pShard->amp1 = amps[1];
                    pShard->isProbDirty = false;
                    pShard->isPhaseDirty = false;
                    pShard->unit = NULL;
                    pShard->mapped = 0;
                    if (doNormalize) {
                        pShard->ClampAmps(amplitudeFloor);
                    }
                }
            }
        } else {
            shard.unit = shards[decomposedUnits[unit]].unit;
        }

        if (dest) {
            dest->shards[i] = shard;
        }
    }

    /* Find the rest of the qubits. */
    for (auto&& shard : shards) {
        auto subunit = subunits.find(shard.unit);
        if (subunit != subunits.end() &&
            shard.mapped >= (shards[decomposedUnits[shard.unit]].mapped + subunit->second)) {
            shard.mapped -= subunit->second;
        }
    }

    shards.erase(start, start + length);
    SetQubitCount(qubitCount - length);
}

void QUnit::Decompose(bitLenInt start, QUnitPtr dest) { Detach(start, dest->GetQubitCount(), dest); }

void QUnit::Dispose(bitLenInt start, bitLenInt length) { Detach(start, length, nullptr); }

// The optimization of this method is redundant with other optimizations in QUnit.
void QUnit::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm) { Detach(start, length, nullptr); }

QInterfacePtr QUnit::EntangleInCurrentBasis(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    for (auto bit = first; bit < last; bit++) {
        EndEmulation(**bit);
    }

    std::vector<QInterfacePtr> units;
    units.reserve((int)(last - first));

    QInterfacePtr unit1 = shards[**first].unit;
    std::map<QInterfacePtr, bool> found;

    /* Walk through all of the supplied bits and create a unique list to compose. */
    for (auto bit = first; bit < last; bit++) {
        if (found.find(shards[**bit].unit) == found.end()) {
            found[shards[**bit].unit] = true;
            units.push_back(shards[**bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    while (units.size() > 1U) {
        // Work odd unit into collapse sequence:
        if (units.size() & 1U) {
            QInterfacePtr consumed = units[1];
            bitLenInt offset = unit1->Compose(consumed);
            units.erase(units.begin() + 1U);

            for (auto&& shard : shards) {
                if (shard.unit == consumed) {
                    shard.mapped += offset;
                    shard.unit = unit1;
                }
            }
        }

        std::vector<QInterfacePtr> nUnits;
        std::map<QInterfacePtr, bitLenInt> offsets;
        std::map<QInterfacePtr, QInterfacePtr> offsetPartners;

        for (size_t i = 0; i < units.size(); i += 2) {
            QInterfacePtr retained = units[i];
            QInterfacePtr consumed = units[i + 1U];
            nUnits.push_back(retained);
            offsets[consumed] = retained->Compose(consumed);
            offsetPartners[consumed] = retained;
        }

        /* Since each unit will be collapsed in-order, one set of bits at a time. */
        for (auto&& shard : shards) {
            auto search = offsets.find(shard.unit);
            if (search != offsets.end()) {
                shard.mapped += search->second;
                shard.unit = offsetPartners[shard.unit];
            }
        }

        units = nUnits;
    }

    /* Change the source parameters to the correct newly mapped bit indexes. */
    for (auto bit = first; bit < last; bit++) {
        **bit = shards[**bit].mapped;
    }

    return unit1;
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt> bits)
{
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(bits.size());
    for (bitLenInt i = 0; i < (bitLenInt)ebits.size(); i++) {
        ebits[i] = &bits[i];
    }

    return Entangle(ebits);
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt*> bits)
{
    for (bitLenInt i = 0; i < (bitLenInt)bits.size(); i++) {
        ToPermBasis(*(bits[i]));
    }
    return EntangleInCurrentBasis(bits.begin(), bits.end());
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start, bitLenInt length, bool isForProb)
{
    if (isForProb) {
        ToPermBasisProb(start, length);
    } else {
        ToPermBasis(start, length);
    }

    if (length == 1) {
        EndEmulation(start);
        return shards[start].unit;
    }

    std::vector<bitLenInt> bits(length);
    std::vector<bitLenInt*> ebits(length);
    for (bitLenInt i = 0; i < length; i++) {
        bits[i] = i + start;
        ebits[i] = &bits[i];
    }

    QInterfacePtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
    OrderContiguous(shards[start].unit);
    return toRet;
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2)
{
    ToPermBasis(start1, length1);
    ToPermBasis(start2, length2);

    std::vector<bitLenInt> bits(length1 + length2);
    std::vector<bitLenInt*> ebits(length1 + length2);

    if (start2 < start1) {
        std::swap(start1, start2);
        std::swap(length1, length2);
    }

    for (bitLenInt i = 0; i < length1; i++) {
        bits[i] = i + start1;
        ebits[i] = &bits[i];
    }

    for (bitLenInt i = 0; i < length2; i++) {
        bits[i + length1] = i + start2;
        ebits[i + length1] = &bits[i + length1];
    }

    QInterfacePtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
    OrderContiguous(shards[start1].unit);
    return toRet;
}

QInterfacePtr QUnit::EntangleRange(
    bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3)
{
    ToPermBasis(start1, length1);
    ToPermBasis(start2, length2);
    ToPermBasis(start3, length3);

    std::vector<bitLenInt> bits(length1 + length2 + length3);
    std::vector<bitLenInt*> ebits(length1 + length2 + length3);

    if (start2 < start1) {
        std::swap(start1, start2);
        std::swap(length1, length2);
    }

    if (start3 < start1) {
        std::swap(start1, start3);
        std::swap(length1, length3);
    }

    if (start3 < start2) {
        std::swap(start2, start3);
        std::swap(length2, length3);
    }

    for (bitLenInt i = 0; i < length1; i++) {
        bits[i] = i + start1;
        ebits[i] = &bits[i];
    }

    for (bitLenInt i = 0; i < length2; i++) {
        bits[i + length1] = i + start2;
        ebits[i + length1] = &bits[i + length1];
    }

    for (bitLenInt i = 0; i < length3; i++) {
        bits[i + length1 + length2] = i + start3;
        ebits[i + length1 + length2] = &bits[i + length1 + length2];
    }

    QInterfacePtr toRet = EntangleInCurrentBasis(ebits.begin(), ebits.end());
    OrderContiguous(shards[start1].unit);
    return toRet;
}

bool QUnit::TrySeparateClifford(bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];

    if (!shard.unit || !shard.unit->isClifford() || !shard.unit->TrySeparate(shard.mapped)) {
        return false;
    }

    // If TrySeparate() == true, this bit can be decomposed.
    QInterfacePtr sepUnit = MakeEngine(1, 0);
    shard.unit->Decompose(shard.mapped, sepUnit);

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if ((shard.unit == shards[i].unit) && (shard.mapped < shards[i].mapped)) {
            shards[i].mapped--;
        }
    }
    shard.mapped = 0;
    shard.unit = sepUnit;

    CacheSingleQubitShard(qubit);

    return true;
}

bool QUnit::TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f error_tol)
{
    if (length == 1U) {
        bitLenInt qubit = qubits[0];
        QEngineShard& shard = shards[qubit];

        if (shard.GetQubitCount() == 1U) {
            return true;
        }

        if (BLOCKED_SEPARATE(shard)) {
            return false;
        }

        bitLenInt mapped = shard.mapped;
        QInterfacePtr oUnit = shard.unit;
        QInterfacePtr nUnit = MakeEngine(1U, 0U);
        if (oUnit->TryDecompose(mapped, nUnit, error_tol)) {
            for (bitLenInt i = 0; i < qubitCount; i++) {
                if ((shards[i].unit == oUnit) && (shards[i].mapped > mapped)) {
                    shards[i].mapped--;
                }
            }

            shard.unit = nUnit;
            shard.mapped = 0;
            shard.MakeDirty();
            CacheSingleQubitShard(qubit);

            if (oUnit->GetQubitCount() == 1U) {
                return true;
            }

            for (bitLenInt i = 0; i < qubitCount; i++) {
                if (shard.unit == oUnit) {
                    CacheSingleQubitShard(i);
                    break;
                }
            }

            return true;
        }

        return false;
    }

    std::vector<bitLenInt> q(length);
    std::copy(qubits, qubits + length, q.begin());
    std::sort(q.begin(), q.end());

    // Swap gate is free, so just bring into the form of the contiguous overload.
    for (bitLenInt i = 0; i < length; i++) {
        Swap(i, q[i]);
    }

    QUnitPtr dest = std::dynamic_pointer_cast<QUnit>(std::make_shared<QUnit>(
        engines, length, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, useHostRam));

    bool toRet = TryDecompose(0, dest, error_tol);
    if (toRet) {
        if (length == 1U) {
            dest->CacheSingleQubitShard(0);
        }
        Compose(dest, 0);
    }

    for (bitLenInt i = 0; i < length; i++) {
        Swap(i, q[i]);
    }

    return toRet;
}

bool QUnit::TrySeparate(bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];

    if (shard.GetQubitCount() == 1U) {
        return true;
    }

    if (freezeTrySeparate) {
        return false;
    }

    if (shard.unit && shard.unit->isClifford()) {
        return TrySeparateClifford(qubit);
    }

    if (!shard.isPauliX && !shard.isPauliY) {
        ConvertZToX(qubit);
    } else if (shard.isPauliY) {
        RevertBasisY(qubit);
    }

    real1_f prob;
    real1_f probX = ZERO_R1;
    real1_f probY = ZERO_R1;
    real1_f probZ = ZERO_R1;

    for (bitLenInt i = 0; i < 3; i++) {
        prob = (ONE_R1 / 2) - ProbBase(qubit);

        if (!shard.unit) {
            return true;
        }

        if (!shard.isPauliX && !shard.isPauliY) {
            probZ = prob;
        } else if (shard.isPauliX) {
            probX = prob;
        } else {
            probY = prob;
        }

        if (i >= 2) {
            continue;
        }

        if (!shard.isPauliX && !shard.isPauliY) {
            ConvertZToX(qubit);
        } else if (shard.isPauliX) {
            ConvertXToY(qubit);
        } else {
            ConvertYToZ(qubit);
        }
    }

    real1_f r = sqrt(probZ * probZ + probX * probX + probY * probY);
    if ((ONE_R1 / 2 - r) > separabilityThreshold) {
        return false;
    }

    real1_f inclination = atan2(sqrt(probX * probX + probY * probY), probZ);
    real1_f azimuth = atan2(probY, probX);

    if (std::isnan(inclination) || std::isinf(inclination) || std::isnan(azimuth) || std::isinf(azimuth)) {
        return false;
    }

    RevertBasis1Qb(qubit);

    shard.unit->IAI(shard.mapped, azimuth, inclination);
    shard.isProbDirty = true;

    prob = (ONE_R1 / 2) - ProbBase(qubit);
    if (!shard.unit) {
        ShardAI(shard, azimuth, inclination);
        return true;
    }

    bool value = prob < ZERO_R1;
    if (value) {
        prob = -prob;
    }
    if ((ONE_R1 / 2 - prob) <= separabilityThreshold) {
        SeparateBit(value, qubit);
        ShardAI(shard, azimuth, inclination);
        return true;
    }

    shard.unit->AI(shard.mapped, azimuth, inclination);

    return false;
}

bool QUnit::TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
{
    // If either shard separates as a single bit, there's no point in checking for entanglement.
    bool isShard1Sep = TrySeparate(qubit1);
    bool isShard2Sep = TrySeparate(qubit2);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (isShard1Sep || isShard2Sep || shard1.unit != shard2.unit) {
        // Both shards have non-null units, and we've tried everything, if they're not the same unit.
        return isShard1Sep && isShard2Sep;
    }

    if (freezeTrySeparate || freezeBasis2Qb) {
        return false;
    }

    // Both shards are in the same unit.
    if (shard1.unit->isClifford() && !shard1.unit->TrySeparate(shard1.mapped, shard2.mapped)) {
        return false;
    }

    real1_f prob1 = ProbBase(qubit1) - ONE_R1 / 2;
    real1_f prob2 = ProbBase(qubit2) - ONE_R1 / 2;
    if ((abs(prob1) > separabilityThreshold) || (abs(prob2) > separabilityThreshold)) {
        // Not worth attempting further.
        return false;
    }
    real1_f probL2I = (prob1 * prob1) + (prob2 * prob2);

    const bool isApproxSep = (separabilityThreshold > FP_NORM_EPSILON);
    const bool is2Qubit = (shard1.unit->GetQubitCount() == 2U) && !isApproxSep;
    const bool wasReactiveSeparate = isReactiveSeparate;
    isReactiveSeparate = true;

    // Try a maximally disentangling operation, in 3 bases.
    RevertBasis1Qb(qubit1);
    RevertBasis1Qb(qubit2);

    // "Kick up" the one possible bit of entanglement entropy into a 2-qubit buffer.
    freezeTrySeparate = true;
    CNOT(qubit1, qubit2);
    shard1.unit->CNOT(shard1.mapped, shard2.mapped);
    freezeTrySeparate = false;

    isShard1Sep = TrySeparate(qubit1);
    if (!is2Qubit) {
        isShard2Sep = TrySeparate(qubit2);
    }
    if (isShard1Sep || isShard2Sep) {
        isReactiveSeparate = wasReactiveSeparate;
        return isShard1Sep && isShard2Sep;
    }

    prob1 = ProbBase(qubit1) - ONE_R1 / 2;
    prob2 = ProbBase(qubit2) - ONE_R1 / 2;
    if ((abs(prob1) > separabilityThreshold) || (abs(prob2) > separabilityThreshold)) {
        // Not worth attempting further.
        isReactiveSeparate = wasReactiveSeparate;
        return false;
    }
    real1_f probL2X = (prob1 * prob1) + (prob2 * prob2);

    bitLenInt control[1] = { shard1.mapped };

    // Revert first basis and try again, in second basis.
    RevertBasis1Qb(qubit1);
    RevertBasis1Qb(qubit2);
    freezeTrySeparate = true;
    CNOT(qubit1, qubit2);
    shard1.unit->MCPhase(control, 1U, -I_CMPLX, I_CMPLX, shard2.mapped);
    CY(qubit1, qubit2);
    freezeTrySeparate = false;

    isShard1Sep = TrySeparate(qubit1);
    if (!is2Qubit) {
        isShard2Sep = TrySeparate(qubit2);
    }
    if (isShard1Sep || isShard2Sep) {
        isReactiveSeparate = wasReactiveSeparate;
        return isShard1Sep && isShard2Sep;
    }

    prob1 = ProbBase(qubit1) - ONE_R1 / 2;
    prob2 = ProbBase(qubit2) - ONE_R1 / 2;
    if ((abs(prob1) > separabilityThreshold) || (abs(prob2) > separabilityThreshold)) {
        // Not worth attempting further.
        isReactiveSeparate = wasReactiveSeparate;
        return false;
    }
    real1_f probL2Y = (prob1 * prob1) + (prob2 * prob2);

    // Revert second basis and try again, in third basis.
    RevertBasis1Qb(qubit1);
    RevertBasis1Qb(qubit2);
    freezeTrySeparate = true;
    CY(qubit1, qubit2);
    shard1.unit->MCInvert(control, 1U, -I_CMPLX, -I_CMPLX, shard2.mapped);
    CZ(qubit1, qubit2);
    freezeTrySeparate = false;

    isShard1Sep = TrySeparate(qubit1);
    if (!is2Qubit) {
        isShard2Sep = TrySeparate(qubit2);
    }
    if (!isApproxSep || isShard1Sep || isShard2Sep) {
        isReactiveSeparate = wasReactiveSeparate;
        return isShard1Sep && isShard2Sep;
    }
    real1_f probL2Z = (prob1 * prob1) + (prob2 * prob2);

    isReactiveSeparate = wasReactiveSeparate;

    // If reactively searching for separability, leave in the state closest to any Bloch sphere poles among Bell basis
    // states.
    // Ordered per Pauli enum definition
    std::vector<real1_f> probL2 = { probL2I, probL2X, probL2Z, probL2Y };
    Pauli bestBasis = (Pauli)std::distance(probL2.begin(), std::max_element(probL2.begin(), probL2.end()));

    if (bestBasis == PauliZ) {
        return false;
    }

    RevertBasis1Qb(qubit1);
    RevertBasis1Qb(qubit2);
    freezeTrySeparate = true;
    CZ(qubit1, qubit2);

    if (isApproxSep) {
        if (bestBasis == PauliX) {
            shard1.unit->MCInvert(control, 1U, -ONE_CMPLX, ONE_CMPLX, shard2.mapped);
            CNOT(qubit1, qubit2);
        } else if (bestBasis == PauliY) {
            shard1.unit->MCInvert(control, 1U, I_CMPLX, I_CMPLX, shard2.mapped);
            CY(qubit1, qubit2);
        }
        // else - Identity
    }

    freezeTrySeparate = false;

    if (isApproxSep) {
        isShard1Sep = TrySeparate(qubit1);
        isShard2Sep = TrySeparate(qubit2);
    }

    return isShard1Sep && isShard2Sep;
}

void QUnit::OrderContiguous(QInterfacePtr unit)
{
    /* Before we call OrderContinguous, when we are cohering lists of shards, we should always proactively sort the
     * order in which we compose qubits into a single engine. This is a cheap way to reduce the need for costly qubit
     * swap gates, later. */

    if (!unit || (unit->GetQubitCount() == 1)) {
        return;
    }

    /* Create a sortable collection of all of the bits that are in the unit. */
    std::vector<QSortEntry> bits(unit->GetQubitCount());

    bitLenInt j = 0;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (shards[i].unit == unit) {
            bits[j].mapped = shards[i].mapped;
            bits[j].bit = i;
            j++;
        }
    }

    SortUnit(unit, bits, 0, bits.size() - 1);
}

/* Sort a container of bits, calling Swap() on each. */
void QUnit::SortUnit(QInterfacePtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high)
{
    bitLenInt i = low, j = high;
    if (i == (j - 1)) {
        if (bits[j] < bits[i]) {
            unit->Swap(bits[i].mapped, bits[j].mapped); /* Change the location in the QE itself. */
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped); /* Change the global mapping. */
            std::swap(bits[i].mapped, bits[j].mapped); /* Change the contents of the sorting array. */
        }
        return;
    }
    QSortEntry pivot = bits[(low + high) / 2];

    while (i <= j) {
        while (bits[i] < pivot) {
            i++;
        }
        while (bits[j] > pivot) {
            j--;
        }
        if (i < j) {
            unit->Swap(bits[i].mapped, bits[j].mapped); /* Change the location in the QE itself. */
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped); /* Change the global mapping. */
            std::swap(bits[i].mapped, bits[j].mapped); /* Change the contents of the sorting array. */
            i++;
            j--;
        } else if (i == j) {
            i++;
            j--;
        }
    }
    if (low < j) {
        SortUnit(unit, bits, low, j);
    }
    if (i < high) {
        SortUnit(unit, bits, i, high);
    }
}

/// Check if the qubit at "qubitIndex" has a cached probability indicating that it is in a permutation basis eigenstate,
/// for optimization.
bool QUnit::CheckBitPermutation(bitLenInt qubitIndex, bool inCurrentBasis)
{
    if (!inCurrentBasis) {
        ToPermBasisProb(qubitIndex);
    }
    QEngineShard& shard = shards[qubitIndex];

    return UNSAFE_CACHED_ZERO_OR_ONE(shard);
}

/// Check if all qubits in the range have cached probabilities indicating that they are in permutation basis
/// eigenstates, for optimization.
bool QUnit::CheckBitsPermutation(bitLenInt start, bitLenInt length, bool inCurrentBasis)
{
    // Certain optimizations become obvious, if all bits in a range are in permutation basis eigenstates.
    // Then, operations can often be treated as classical, instead of quantum.

    if (!inCurrentBasis) {
        ToPermBasisProb(start, length);
    }

    for (bitLenInt i = 0; i < length; i++) {
        QEngineShard& shard = shards[start + i];
        if (!UNSAFE_CACHED_ZERO_OR_ONE(shard)) {
            return false;
        }
    }
    return true;
}

/// Assuming all bits in the range are in cached |0>/|1> eigenstates, read the unsigned integer value of the range.
bitCapInt QUnit::GetCachedPermutation(bitLenInt start, bitLenInt length)
{
    bitCapInt res = 0U;
    for (bitLenInt i = 0; i < length; i++) {
        if (SHARD_STATE(shards[start + i])) {
            res |= pow2(i);
        }
    }
    return res;
}

bitCapInt QUnit::GetCachedPermutation(const bitLenInt* bitArray, bitLenInt length)
{
    bitCapInt res = 0U;
    for (bitLenInt i = 0; i < length; i++) {
        if (SHARD_STATE(shards[bitArray[i]])) {
            res |= pow2(i);
        }
    }
    return res;
}

bool QUnit::CheckBitsPlus(bitLenInt qubitIndex, bitLenInt length)
{
    bool isHBasis = true;
    for (bitLenInt i = 0; i < length; i++) {
        QEngineShard& shard = shards[qubitIndex + i];
        if (!CACHED_PLUS(shard)) {
            isHBasis = false;
            break;
        }
    }

    return isHBasis;
}

real1_f QUnit::ProbBase(bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];

    if (!shard.isProbDirty) {
        real1_f prob = clampProb(norm(shard.amp1));
        if (shard.unit) {
            if (IS_NORM_0(shard.amp1)) {
                SeparateBit(false, qubit);
            } else if (IS_NORM_0(shard.amp0)) {
                SeparateBit(true, qubit);
            }
        }

        return prob;
    }

    shard.isProbDirty = false;

    QInterfacePtr unit = shard.unit;
    bitLenInt mapped = shard.mapped;
    real1_f prob = unit->Prob(mapped);
    shard.amp1 = complex((real1)sqrt(prob), ZERO_R1);
    shard.amp0 = complex((real1)sqrt(ONE_R1 - prob), ZERO_R1);

    if (IS_NORM_0(shard.amp1)) {
        SeparateBit(false, qubit);
    } else if (IS_NORM_0(shard.amp0)) {
        SeparateBit(true, qubit);
    }

    return prob;
}

void QUnit::CacheSingleQubitShard(bitLenInt target)
{
    RevertBasis1Qb(target);
    QEngineShard& shard = shards[target];

    if (!shard.unit) {
        return;
    }

    complex amps[2];
    shard.unit->GetQuantumState(amps);
    if (IS_AMP_0(amps[0] - amps[1])) {
        shard.isPauliX = true;
        shard.isPauliY = false;
        amps[0] = amps[0] / abs(amps[0]);
        amps[1] = ZERO_CMPLX;
    } else if (IS_AMP_0(amps[0] + amps[1])) {
        shard.isPauliX = true;
        shard.isPauliY = false;
        amps[1] = amps[0] / abs(amps[0]);
        amps[0] = ZERO_CMPLX;
    } else if (IS_AMP_0((I_CMPLX * amps[0]) - amps[1])) {
        shard.isPauliX = false;
        shard.isPauliY = true;
        amps[0] = amps[0] / abs(amps[0]);
        amps[1] = ZERO_CMPLX;
    } else if (IS_AMP_0((I_CMPLX * amps[0]) + amps[1])) {
        shard.isPauliX = false;
        shard.isPauliY = true;
        amps[1] = amps[0] / abs(amps[0]);
        amps[0] = ZERO_CMPLX;
    }
    shard.amp0 = amps[0];
    shard.amp1 = amps[1];
    shard.isProbDirty = false;
    shard.isPhaseDirty = false;
    shard.unit = NULL;
    shard.mapped = 0;
    if (doNormalize) {
        shard.ClampAmps(amplitudeFloor);
    }
}

real1_f QUnit::Prob(bitLenInt qubit)
{
    ToPermBasisProb(qubit);
    return ProbBase(qubit);
}

real1_f QUnit::ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset)
{
    if ((length == 1U) || (shards[0].GetQubitCount() != qubitCount)) {
        return QInterface::ExpectationBitsAll(bits, length, offset);
    }

    ToPermBasisProb();
    OrderContiguous(shards[0].unit);

    return shards[0].unit->ExpectationBitsAll(bits, length, offset);
}

real1_f QUnit::ProbAll(bitCapInt perm) { return clampProb(norm(GetAmplitudeOrProb(perm, true))); }

real1_f QUnit::ProbParity(bitCapInt mask)
{
    // If no bits in mask:
    if (!mask) {
        return ZERO_R1;
    }

    if (!(mask & (mask - ONE_BCI))) {
        return Prob(log2(mask));
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));

        RevertBasis2Qb(qIndices.back(), ONLY_INVERT, ONLY_TARGETS);

        QEngineShard& shard = shards[qIndices.back()];
        if (shard.unit && QUEUED_PHASE(shard)) {
            RevertBasis1Qb(qIndices.back());
        }
    }

    std::map<QInterfacePtr, bitCapInt> units;
    real1 oddChance = ZERO_R1;
    real1 nOddChance;
    for (bitLenInt i = 0; i < (bitLenInt)qIndices.size(); i++) {
        QEngineShard& shard = shards[qIndices[i]];
        if (!(shard.unit)) {
            nOddChance =
                (shard.isPauliX || shard.isPauliY) ? norm(SQRT1_2_R1 * (shard.amp0 - shard.amp1)) : shard.Prob();
            oddChance = (oddChance * (ONE_R1 - nOddChance)) + ((ONE_R1 - oddChance) * nOddChance);
            continue;
        }

        RevertBasis1Qb(qIndices[i]);

        units[shard.unit] |= pow2(shard.mapped);
    }

    if (qIndices.size() == 0) {
        return oddChance;
    }

    std::map<QInterfacePtr, bitCapInt>::iterator unit;
    for (unit = units.begin(); unit != units.end(); unit++) {
        nOddChance = unit->first->ProbParity(unit->second);
        oddChance = (oddChance * (ONE_R1 - nOddChance)) + ((ONE_R1 - oddChance) * nOddChance);
    }

    return oddChance;
}

bool QUnit::ForceMParity(bitCapInt mask, bool result, bool doForce)
{
    // If no bits in mask:
    if (!mask) {
        return false;
    }

    if (!(mask & (mask - ONE_BCI))) {
        return ForceM(log2(mask), result, doForce);
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
        ToPermBasisProb(qIndices.back());
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (bitLenInt i = 0; i < (bitLenInt)qIndices.size(); i++) {
        QEngineShard& shard = shards[qIndices[i]];

        if (UNSAFE_CACHED_ZERO(shard)) {
            continue;
        }

        if (UNSAFE_CACHED_ONE(shard)) {
            flipResult = !flipResult;
            continue;
        }

        eIndices.push_back(qIndices[i]);
    }

    if (eIndices.size() == 0) {
        return flipResult;
    }

    if (eIndices.size() == 1U) {
        return flipResult ^ ForceM(eIndices[0], result ^ flipResult, doForce);
    }

    QInterfacePtr unit = Entangle(eIndices);

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (shards[i].unit == unit) {
            shards[i].MakeDirty();
        }
    }

    bitCapInt mappedMask = 0;
    for (bitLenInt i = 0; i < (bitLenInt)eIndices.size(); i++) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    return flipResult ^ (unit->ForceMParity(mappedMask, result ^ flipResult, doForce));
}

void QUnit::PhaseParity(real1 radians, bitCapInt mask)
{
    // If no bits in mask:
    if (!mask) {
        return;
    }

    complex phaseFac = complex((real1)cos(radians / 2), (real1)sin(radians / 2));

    if (!(mask & (mask - ONE_BCI))) {
        Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        return;
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
        ToPermBasisProb(qIndices.back());
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (bitLenInt i = 0; i < (bitLenInt)qIndices.size(); i++) {
        QEngineShard& shard = shards[qIndices[i]];

        if (UNSAFE_CACHED_ZERO(shard)) {
            continue;
        }

        if (UNSAFE_CACHED_ONE(shard)) {
            flipResult = !flipResult;
            continue;
        }

        eIndices.push_back(qIndices[i]);
    }

    if (eIndices.size() == 0) {
        if (flipResult) {
            Phase(phaseFac, phaseFac, 0U);
        } else {
            Phase(ONE_CMPLX / phaseFac, ONE_CMPLX / phaseFac, 0U);
        }
        return;
    }

    if (eIndices.size() == 1U) {
        if (flipResult) {
            Phase(phaseFac, ONE_CMPLX / phaseFac, log2(mask));
        } else {
            Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        }
        return;
    }

    QInterfacePtr unit = Entangle(eIndices);

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (shards[i].unit == unit) {
            shards[i].MakeDirty();
        }
    }

    bitCapInt mappedMask = 0;
    for (bitLenInt i = 0; i < (bitLenInt)eIndices.size(); i++) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    unit->PhaseParity(flipResult ? -radians : radians, mappedMask);
}

bool QUnit::SeparateBit(bool value, bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];
    QInterfacePtr unit = shard.unit;
    bitLenInt mapped = shard.mapped;

    if (unit->isClifford() && !unit->TrySeparate(mapped)) {
        // This conditional coaxes the unit into separable form, so this should never actually happen.
        return false;
    }

    real1_f prob = shard.Prob();

    shard.unit = NULL;
    shard.mapped = 0;
    shard.isProbDirty = false;
    shard.isPhaseDirty = false;
    shard.amp0 = value ? ZERO_CMPLX : GetNonunitaryPhase();
    shard.amp1 = value ? GetNonunitaryPhase() : ZERO_CMPLX;

    if (!unit || (unit->GetQubitCount() == 1U)) {
        return true;
    }

    if (prob <= FP_NORM_EPSILON) {
        value = false;
    } else if ((ONE_R1 - prob) <= FP_NORM_EPSILON) {
        value = true;
    }

    unit->Dispose(mapped, 1, value ? ONE_BCI : 0);
    if (!unit->isBinaryDecisionTree() && (((ONE_R1 / 2) - abs((ONE_R1 / 2) - prob)) > FP_NORM_EPSILON)) {
        unit->UpdateRunningNorm();
        if (!doNormalize) {
            unit->NormalizeState();
        }
        for (auto&& s : shards) {
            if (s.unit == unit) {
                s.MakeDirty();
            }
        }
    }

    /* Update the mappings. */
    for (auto&& s : shards) {
        if ((s.unit == unit) && (s.mapped > mapped)) {
            s.mapped--;
        }
    }

    if (unit->GetQubitCount() != 1) {
        return true;
    }

    bitLenInt partnerIndex;
    for (partnerIndex = 0; partnerIndex < qubitCount; partnerIndex++) {
        QEngineShard& partnerShard = shards[partnerIndex];
        if (unit == partnerShard.unit) {
            break;
        }
    }

    CacheSingleQubitShard(partnerIndex);

    return true;
}

bool QUnit::ForceM(bitLenInt qubit, bool res, bool doForce, bool doApply)
{
    if (doApply) {
        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT, ONLY_TARGETS);
    } else {
        ToPermBasisMeasure(qubit);
    }

    QEngineShard& shard = shards[qubit];

    bool result;
    if (!shard.isProbDirty && (!shard.unit || (!shard.unit->isClifford() && !shard.unit->isBinaryDecisionTree()))) {
        real1_f prob = norm(shard.amp1);
        if (doForce) {
            result = res;
        } else if (prob >= ONE_R1) {
            result = true;
        } else if (prob <= ZERO_R1) {
            result = false;
        } else {
            result = (Rand() <= prob);
        }
    } else {
        // Binary decision tree HAS to be collapsed into the right state before Decompose()/Dispose().
        result = shard.unit->ForceM(shard.mapped, res, doForce, doApply);
    }

    if (!doApply) {
        return result;
    }

    shard.isProbDirty = false;
    shard.isPhaseDirty = false;
    shard.amp0 = result ? ZERO_CMPLX : GetNonunitaryPhase();
    shard.amp1 = result ? GetNonunitaryPhase() : ZERO_CMPLX;

    if (shard.GetQubitCount() == 1U) {
        shard.unit = NULL;
        shard.mapped = 0;
        if (result) {
            Flush1Eigenstate(qubit);
        } else {
            Flush0Eigenstate(qubit);
        }
        return result;
    }

    // This is critical: it's the "nonlocal correlation" of "wave function collapse".
    if (shard.unit) {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if ((i != qubit) && shards[i].unit == shard.unit) {
                shards[i].MakeDirty();
            }
        }
        SeparateBit(result, qubit);
    }

    if (result) {
        Flush1Eigenstate(qubit);
    } else {
        Flush0Eigenstate(qubit);
    }

    return result;
}

bitCapInt QUnit::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    if (!doForce && doApply && (length == qubitCount)) {
        return MAll();
    }

    // This will discard all buffered gates that don't affect Z basis probability,
    // so it's safe to call ToPermBasis() without performance penalty, afterward.
    if (!doApply) {
        ToPermBasisMeasure(start, length);
    }

    return QInterface::ForceMReg(start, length, result, doForce, doApply);
}

bitCapInt QUnit::MAll()
{
    for (bitLenInt i = 0; i < qubitCount; i++) {
        RevertBasis1Qb(i);
    }
    for (bitLenInt i = 0; i < qubitCount; i++) {
        QEngineShard& shard = shards[i];
        shard.ClearInvertPhase();
        shard.DumpPhaseBuffers();
    }

    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QEngineShard shard = shards[i];
        if (shard.IsInvertControl() && shard.unit && shard.unit->isClifford()) {
            // Clifford might become ket during measurement
            ToPermBasisMeasure(i);
        }
    }

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (shards[i].IsInvertControl()) {
            // Measurement commutes with control
            M(i);
        }
    }

    bitCapInt toRet = 0;

    units.clear();
    std::map<QInterfacePtr, bitCapInt> partResult;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (!toFind) {
            real1_f prob = norm(shards[i].amp1);
            if ((prob >= ONE_R1) || ((prob > ZERO_R1) && (Rand() <= prob))) {
                shards[i].amp0 = ZERO_CMPLX;
                shards[i].amp1 = GetNonunitaryPhase();
                toRet |= pow2(i);
            } else {
                shards[i].amp0 = GetNonunitaryPhase();
                shards[i].amp1 = ZERO_CMPLX;
            }
        } else if (!toFind->isClifford() && !toFind->isBinaryDecisionTree()) {
            if (M(i)) {
                toRet |= pow2(i);
            }
        } else {
            if (std::find(units.begin(), units.end(), toFind) == units.end()) {
                units.push_back(toFind);
                partResult[toFind] = toFind->MAll();
            }
            toRet |= (((partResult[toFind] >> shards[i].mapped) & 1U) << i);
        }
    }

    SetPermutation(toRet);

    return toRet;
}

std::map<bitCapInt, int> QUnit::MultiShotMeasureMask(const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    ToPermBasisProb();

    bitLenInt index;
    std::vector<bitLenInt> qIndices(qPowerCount);
    std::map<bitLenInt, bitCapInt> iQPowers;
    for (bitLenInt i = 0U; i < qPowerCount; i++) {
        index = log2(qPowers[i]);
        qIndices[i] = index;
        iQPowers[index] = pow2(i);
    }

    std::map<QInterfacePtr, std::vector<bitCapInt>> subQPowers;
    std::map<QInterfacePtr, std::vector<bitCapInt>> subIQPowers;
    std::vector<bitLenInt> singleBits;

    for (bitLenInt i = 0U; i < qPowerCount; i++) {
        index = qIndices[i];
        QEngineShard& shard = shards[index];

        if (!shard.unit) {
            singleBits.push_back(index);
            continue;
        }

        subQPowers[shard.unit].push_back(pow2(shard.mapped));
        subIQPowers[shard.unit].push_back(iQPowers[index]);
    }

    std::map<bitCapInt, int> combinedResults;
    combinedResults[0U] = (int)shots;

    for (auto subQPowersIt = subQPowers.begin(); subQPowersIt != subQPowers.end(); subQPowersIt++) {
        QInterfacePtr unit = subQPowersIt->first;
        std::map<bitCapInt, int> unitResults =
            unit->MultiShotMeasureMask(&(subQPowersIt->second[0]), subQPowersIt->second.size(), shots);
        std::map<bitCapInt, int> topLevelResults;
        for (auto mapIter = unitResults.begin(); mapIter != unitResults.end(); mapIter++) {
            bitCapInt mask = 0U;
            for (bitLenInt i = 0U; i < (bitLenInt)subQPowersIt->second.size(); i++) {
                if ((mapIter->first >> i) & 1U) {
                    mask |= subIQPowers[unit][i];
                }
            }
            topLevelResults[mask] = mapIter->second;
        }
        // Release unitResults memory:
        unitResults = std::map<bitCapInt, int>();

        // If either map is fully |0>, nothing changes (after the swap).
        if ((topLevelResults.begin()->first == 0) && (topLevelResults[0] == (int)shots)) {
            continue;
        }
        if ((combinedResults.begin()->first == 0) && (combinedResults[0] == (int)shots)) {
            std::swap(topLevelResults, combinedResults);
            continue;
        }

        // Swap if needed, so topLevelResults.size() is smaller.
        if (combinedResults.size() < topLevelResults.size()) {
            std::swap(topLevelResults, combinedResults);
        }
        // (Since swapped...)

        std::map<bitCapInt, int> nCombinedResults;

        // If either map has exactly 1 key, (therefore with `shots` value,) pass it through without a "shuffle."
        if (topLevelResults.size() == 1U) {
            auto pickIter = topLevelResults.begin();
            for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); mapIter++) {
                nCombinedResults[mapIter->first | pickIter->first] = mapIter->second;
            }
            combinedResults = nCombinedResults;
            continue;
        }

        // ... Otherwise, we've committed to simulating a random pairing selection from either side, (but
        // `topLevelResults` has fewer or the same count of keys).
        int shotsLeft = shots;
        for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); mapIter++) {
            for (int shot = 0; shot < mapIter->second; shot++) {
                int pick = (int)(shotsLeft * Rand());
                if (shotsLeft <= pick) {
                    pick = shotsLeft - 1;
                }
                shotsLeft--;

                auto pickIter = topLevelResults.begin();
                int count = pickIter->second;
                while (pick > count) {
                    pickIter++;
                    count += pickIter->second;
                }

                nCombinedResults[mapIter->first | pickIter->first]++;

                pickIter->second--;
                if (pickIter->second == 0) {
                    topLevelResults.erase(pickIter);
                }
            }
        }
        combinedResults = nCombinedResults;
    }

    for (bitLenInt i = 0U; i < (bitLenInt)singleBits.size(); i++) {
        index = singleBits[i];

        real1_f prob = clampProb(norm(shards[index].amp1));
        if (prob == ZERO_R1) {
            continue;
        }

        std::map<bitCapInt, int> nCombinedResults;
        if (prob == ONE_R1) {
            for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); mapIter++) {
                nCombinedResults[mapIter->first | iQPowers[index]] = mapIter->second;
            }
        } else {
            for (auto mapIter = combinedResults.begin(); mapIter != combinedResults.end(); mapIter++) {
                bitCapInt zeroPerm = mapIter->first;
                bitCapInt onePerm = mapIter->first | iQPowers[index];
                for (int shot = 0; shot < mapIter->second; shot++) {
                    if (Rand() > prob) {
                        nCombinedResults[zeroPerm]++;
                    } else {
                        nCombinedResults[onePerm]++;
                    }
                }
            }
        }
        combinedResults = nCombinedResults;
    }

    return combinedResults;
}

void QUnit::MultiShotMeasureMask(const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots, unsigned* shotsArray)
{
    if (!shots) {
        return;
    }

    ToPermBasisProb();

    QInterfacePtr unit = shards[log2(qPowers[0])].unit;
    if (unit) {
        std::unique_ptr<bitCapInt[]> mappedIndices(new bitCapInt[qPowerCount]);
        for (bitLenInt j = 0; j < qubitCount; j++) {
            if (qPowers[0] == pow2(j)) {
                mappedIndices[0] = pow2(shards[j].mapped);
                break;
            }
        }
        for (bitLenInt i = 1U; i < qPowerCount; i++) {
            if (unit != shards[log2(qPowers[i])].unit) {
                unit = NULL;
                break;
            }
            for (bitLenInt j = 0; j < qubitCount; j++) {
                if (qPowers[i] == pow2(j)) {
                    mappedIndices[i] = pow2(shards[j].mapped);
                    break;
                }
            }
        }

        if (unit) {
            unit->MultiShotMeasureMask(mappedIndices.get(), qPowerCount, shots, shotsArray);
            return;
        }
    }

    std::map<bitCapInt, int> results = MultiShotMeasureMask(qPowers, qPowerCount, shots);

    size_t j = 0;
    std::map<bitCapInt, int>::iterator it = results.begin();
    while (it != results.end() && (j < shots)) {
        for (int i = 0; i < it->second; i++) {
            shotsArray[j] = (unsigned)it->first;
            j++;
        }

        it++;
    }
}

/// Set register bits to given permutation
void QUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    MReg(start, length);

    for (bitLenInt i = 0; i < length; i++) {
        bool bitState = ((value >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards[i + start] = QEngineShard(bitState, GetNonunitaryPhase());
    }
}

void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    // Simply swap the bit mapping.
    shards.swap(qubit1, qubit2);
}

void QUnit::ISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (IS_SAME_UNIT(shard1, shard2)) {
        Entangle({ qubit1, qubit2 })->ISwap(shard1.mapped, shard2.mapped);
        shard1.MakeDirty();
        shard2.MakeDirty();
        return;
    }

    bitLenInt control[1] = { qubit1 };
    MCPhase(control, 1U, I_CMPLX, ONE_CMPLX, qubit2);
    control[0] = qubit2;
    MCPhase(control, 1U, I_CMPLX, ONE_CMPLX, qubit1);

    // Simply swap the bit mapping.
    shards.swap(qubit1, qubit2);
}

void QUnit::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    RevertBasis2Qb(qubit1, ONLY_INVERT);
    RevertBasis2Qb(qubit2, ONLY_INVERT);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (UNSAFE_CACHED_ZERO_OR_ONE(shard1) && UNSAFE_CACHED_ZERO_OR_ONE(shard2) &&
        (SHARD_STATE(shard1) == SHARD_STATE(shard2))) {
        // We can avoid dirtying the cache and entangling, since this gate doesn't swap identical classical bits.
        return;
    }

    Entangle({ qubit1, qubit2 })->SqrtSwap(shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();
}

void QUnit::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    RevertBasis2Qb(qubit1, ONLY_INVERT);
    RevertBasis2Qb(qubit2, ONLY_INVERT);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (UNSAFE_CACHED_ZERO_OR_ONE(shard1) && UNSAFE_CACHED_ZERO_OR_ONE(shard2) &&
        (SHARD_STATE(shard1) == SHARD_STATE(shard2))) {
        // We can avoid dirtying the cache and entangling, since this gate doesn't swap identical classical bits.
        return;
    }

    Entangle({ qubit1, qubit2 })->ISqrtSwap(shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();
}

void QUnit::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    bitLenInt controls[1] = { qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if (IS_0_R1(sinTheta)) {
        MCPhase(controls, 1, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    if (IS_1_R1(-sinTheta)) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, 1, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    RevertBasis2Qb(qubit1, ONLY_INVERT);
    RevertBasis2Qb(qubit2, ONLY_INVERT);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (UNSAFE_CACHED_ZERO_OR_ONE(shard1) && UNSAFE_CACHED_ZERO_OR_ONE(shard2) &&
        (SHARD_STATE(shard1) == SHARD_STATE(shard2))) {
        // We can avoid dirtying the cache and entangling, since this gate doesn't swap identical classical bits.
        if (SHARD_STATE(shard1)) {
            MCPhase(controls, 1, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        }
        return;
    }

    Entangle({ qubit1, qubit2 })->FSim(theta, phi, shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();
}

void QUnit::UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask)
{
    // If there are no controls, this is equivalent to the single bit gate.
    if (!controlLen) {
        Mtrx(mtrxs, qubitIndex);
        return;
    }

    std::vector<bitLenInt> trimmedControls;
    std::vector<bitCapInt> skipPowers;
    bitCapInt skipValueMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (!CheckBitPermutation(controls[i])) {
            trimmedControls.push_back(controls[i]);
        } else {
            skipPowers.push_back(pow2(i));
            skipValueMask |= (SHARD_STATE(shards[controls[i]]) ? pow2(i) : 0);
        }
    }

    // If all controls are in eigenstates, we can avoid entangling them.
    if (!trimmedControls.size()) {
        bitCapInt controlPerm = GetCachedPermutation(controls, controlLen);
        complex mtrx[4];
        std::copy(
            mtrxs + (bitCapIntOcl)(controlPerm * 4UL), mtrxs + (bitCapIntOcl)((controlPerm + ONE_BCI) * 4U), mtrx);
        Mtrx(mtrx, qubitIndex);
        return;
    }

    std::vector<bitLenInt> bits(trimmedControls.size() + 1);
    for (bitLenInt i = 0; i < (bitLenInt)trimmedControls.size(); i++) {
        bits[i] = trimmedControls[i];
    }
    bits[trimmedControls.size()] = qubitIndex;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(trimmedControls.size() + 1);
    for (bitLenInt i = 0; i < (bitLenInt)bits.size(); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    std::unique_ptr<bitLenInt[]> mappedControls(new bitLenInt[trimmedControls.size()]);
    for (bitLenInt i = 0; i < (bitLenInt)trimmedControls.size(); i++) {
        mappedControls[i] = shards[trimmedControls[i]].mapped;
        shards[trimmedControls[i]].isPhaseDirty = true;
    }

    unit->UniformlyControlledSingleBit(mappedControls.get(), trimmedControls.size(), shards[qubitIndex].mapped, mtrxs,
        &(skipPowers[0]), skipPowers.size(), skipValueMask);

    shards[qubitIndex].MakeDirty();
}

void QUnit::CUniformParityRZ(const bitLenInt* cControls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
{
    std::vector<bitLenInt> controls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        QEngineShard& shard = shards[cControls[i]];

        if (!CACHED_Z(shard)) {
            // Control becomes entangled
            controls.push_back(cControls[i]);
            continue;
        }

        if (IS_AMP_0(shard.amp1)) {
            // Gate does nothing
            return;
        }

        if (!IS_AMP_0(shard.amp0)) {
            // Control becomes entangled
            controls.push_back(cControls[i]);
        }
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (bitLenInt i = 0; i < (bitLenInt)qIndices.size(); i++) {
        ToPermBasis(qIndices[i]);
        QEngineShard& shard = shards[qIndices[i]];

        if (CACHED_ZERO(shard)) {
            continue;
        }

        if (CACHED_ONE(shard)) {
            flipResult = !flipResult;
            continue;
        }

        eIndices.push_back(qIndices[i]);
    }

    if (eIndices.size() == 0) {
        real1 cosine = (real1)cos(angle);
        real1 sine = (real1)sin(angle);
        complex phaseFac;
        if (flipResult) {
            phaseFac = complex(cosine, sine);
        } else {
            phaseFac = complex(cosine, -sine);
        }
        if (controls.size() == 0) {
            return Phase(phaseFac, phaseFac, 0);
        } else {
            return MCPhase(&(controls[0]), controls.size(), phaseFac, phaseFac, 0);
        }
    }

    if (eIndices.size() == 1U) {
        real1 cosine = (real1)cos(angle);
        real1 sine = (real1)sin(angle);
        complex phaseFac, phaseFacAdj;
        if (flipResult) {
            phaseFac = complex(cosine, -sine);
            phaseFacAdj = complex(cosine, sine);
        } else {
            phaseFac = complex(cosine, sine);
            phaseFacAdj = complex(cosine, -sine);
        }
        if (controls.size() == 0) {
            return Phase(phaseFacAdj, phaseFac, eIndices[0]);
        } else {
            return MCPhase(&(controls[0]), controls.size(), phaseFacAdj, phaseFac, eIndices[0]);
        }
    }

    for (bitLenInt i = 0; i < (bitLenInt)eIndices.size(); i++) {
        shards[eIndices[i]].isPhaseDirty = true;
    }

    QInterfacePtr unit = Entangle(eIndices);

    bitCapInt mappedMask = 0;
    for (bitLenInt i = 0; i < (bitLenInt)eIndices.size(); i++) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    if (controls.size() == 0) {
        unit->UniformParityRZ(mappedMask, flipResult ? -angle : angle);
    } else {
        std::vector<bitLenInt*> ebits(controls.size());
        for (bitLenInt i = 0; i < (bitLenInt)controls.size(); i++) {
            ebits[i] = &controls[i];
        }

        Entangle(ebits);
        unit = Entangle({ controls[0], eIndices[0] });

        std::vector<bitLenInt> controlsMapped(controls.size());
        for (bitLenInt i = 0; i < (bitLenInt)controls.size(); i++) {
            QEngineShard& cShard = shards[controls[i]];
            controlsMapped[i] = cShard.mapped;
            cShard.isPhaseDirty = true;
        }

        unit->CUniformParityRZ(&(controlsMapped[0]), controlsMapped.size(), mappedMask, flipResult ? -angle : angle);
    }
}

void QUnit::H(bitLenInt target)
{
    RevertBasisY(target);
    CommuteH(target);

    QEngineShard& shard = shards[target];
    shard.isPauliX = !shard.isPauliX;
}

void QUnit::S(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    shard.CommutePhase(ONE_CMPLX, I_CMPLX);

    if (UNSAFE_CACHED_ZERO_OR_ONE(shard)) {
        if (SHARD_STATE(shard)) {
            Flush1Eigenstate(target);
        } else {
            Flush0Eigenstate(target);
        }
        return;
    }

    if (shard.isPauliY) {
        shard.isPauliX = true;
        shard.isPauliY = false;
        XBase(target);
        return;
    } else if (shard.isPauliX) {
        shard.isPauliX = false;
        shard.isPauliY = true;
        return;
    }

    if (shard.unit) {
        shard.unit->S(shard.mapped);
    }

    shard.amp1 = I_CMPLX * shard.amp1;
}

void QUnit::IS(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    shard.CommutePhase(ONE_CMPLX, -I_CMPLX);

    if (UNSAFE_CACHED_ZERO_OR_ONE(shard)) {
        if (SHARD_STATE(shard)) {
            Flush1Eigenstate(target);
        } else {
            Flush0Eigenstate(target);
        }
        return;
    }

    if (shard.isPauliY) {
        shard.isPauliX = true;
        shard.isPauliY = false;
        return;
    } else if (shard.isPauliX) {
        shard.isPauliX = false;
        shard.isPauliY = true;
        XBase(target);
        return;
    }

    if (shard.unit) {
        shard.unit->IS(shard.mapped);
    }

    shard.amp1 = -I_CMPLX * shard.amp1;
}

void QUnit::XBase(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->X(shard.mapped);
    }

    std::swap(shard.amp0, shard.amp1);
}

void QUnit::YBase(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->Y(shard.mapped);
    }

    const complex Y0 = shard.amp0;
    shard.amp0 = -I_CMPLX * shard.amp1;
    shard.amp1 = I_CMPLX * Y0;
}

void QUnit::ZBase(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->Z(shard.mapped);
    }

    shard.amp1 = -shard.amp1;
}

void QUnit::X(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    shard.FlipPhaseAnti();

    if (shard.isPauliY) {
        YBase(target);
    } else if (shard.isPauliX) {
        ZBase(target);
    } else {
        XBase(target);
    }
}

void QUnit::Z(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    shard.CommutePhase(ONE_CMPLX, -ONE_CMPLX);

    if (UNSAFE_CACHED_ZERO_OR_ONE(shard)) {
        if (SHARD_STATE(shard)) {
            Flush1Eigenstate(target);
        } else {
            Flush0Eigenstate(target);
        }
        return;
    }

    if (shard.isPauliX || shard.isPauliY) {
        XBase(target);
    } else {
        ZBase(target);
    }
}

void QUnit::TransformX2x2(const complex* mtrxIn, complex* mtrxOut)
{
    mtrxOut[0] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + mtrxIn[1] + mtrxIn[2] + mtrxIn[3]);
    mtrxOut[1] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] - mtrxIn[1] + mtrxIn[2] - mtrxIn[3]);
    mtrxOut[2] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + mtrxIn[1] - mtrxIn[2] - mtrxIn[3]);
    mtrxOut[3] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] - mtrxIn[1] - mtrxIn[2] + mtrxIn[3]);
}

void QUnit::TransformXInvert(complex topRight, complex bottomLeft, complex* mtrxOut)
{
    mtrxOut[0] = (real1)(ONE_R1 / 2) * (complex)(topRight + bottomLeft);
    mtrxOut[1] = (real1)(ONE_R1 / 2) * (complex)(-topRight + bottomLeft);
    mtrxOut[2] = -mtrxOut[1];
    mtrxOut[3] = -mtrxOut[0];
}

void QUnit::TransformY2x2(const complex* mtrxIn, complex* mtrxOut)
{
    mtrxOut[0] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + I_CMPLX * (mtrxIn[1] - mtrxIn[2]) + mtrxIn[3]);
    mtrxOut[1] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] - I_CMPLX * (mtrxIn[1] + mtrxIn[2]) - mtrxIn[3]);
    mtrxOut[2] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + I_CMPLX * (mtrxIn[1] + mtrxIn[2]) - mtrxIn[3]);
    mtrxOut[3] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] - I_CMPLX * (mtrxIn[1] - mtrxIn[2]) + mtrxIn[3]);
}

void QUnit::TransformYInvert(complex topRight, complex bottomLeft, complex* mtrxOut)
{
    mtrxOut[0] = I_CMPLX * (real1)(ONE_R1 / 2) * (complex)(topRight - bottomLeft);
    mtrxOut[1] = I_CMPLX * (real1)(ONE_R1 / 2) * (complex)(-topRight - bottomLeft);
    mtrxOut[2] = -mtrxOut[1];
    mtrxOut[3] = -mtrxOut[0];
}

void QUnit::TransformPhase(complex topLeft, complex bottomRight, complex* mtrxOut)
{
    mtrxOut[0] = (real1)(ONE_R1 / 2) * (complex)(topLeft + bottomRight);
    mtrxOut[1] = (real1)(ONE_R1 / 2) * (complex)(topLeft - bottomRight);
    mtrxOut[2] = mtrxOut[1];
    mtrxOut[3] = mtrxOut[0];
}

#define CTRLED_GEN_WRAP(ctrld, bare, anti)                                                                             \
    ApplyEitherControlled(                                                                                             \
        controls, controlLen, { target }, anti,                                                                        \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {                                               \
            complex trnsMtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                                  \
            if (shards[target].isPauliX) {                                                                             \
                TransformX2x2(mtrx, trnsMtrx);                                                                         \
            } else if (shards[target].isPauliY) {                                                                      \
                TransformY2x2(mtrx, trnsMtrx);                                                                         \
            } else {                                                                                                   \
                std::copy(mtrx, mtrx + 4, trnsMtrx);                                                                   \
            }                                                                                                          \
            unit->ctrld;                                                                                               \
        },                                                                                                             \
        [&]() { bare; });

#define CTRLED_PHASE_INVERT_WRAP(ctrld, ctrldgen, bare, anti, isInvert, top, bottom)                                   \
    ApplyEitherControlled(                                                                                             \
        controls, controlLen, { target }, anti,                                                                        \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {                                               \
            if (shards[target].isPauliX) {                                                                             \
                complex trnsMtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                              \
                if (isInvert) {                                                                                        \
                    TransformXInvert(top, bottom, trnsMtrx);                                                           \
                } else {                                                                                               \
                    TransformPhase(top, bottom, trnsMtrx);                                                             \
                }                                                                                                      \
                unit->ctrldgen;                                                                                        \
            } else if (shards[target].isPauliY) {                                                                      \
                complex trnsMtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                              \
                if (isInvert) {                                                                                        \
                    TransformYInvert(top, bottom, trnsMtrx);                                                           \
                } else {                                                                                               \
                    TransformPhase(top, bottom, trnsMtrx);                                                             \
                }                                                                                                      \
                unit->ctrldgen;                                                                                        \
            } else {                                                                                                   \
                unit->ctrld;                                                                                           \
            }                                                                                                          \
        },                                                                                                             \
        [&]() { bare; }, !isInvert, isInvert);

#define CTRLED_SWAP_WRAP(ctrld, bare, anti)                                                                            \
    if (qubit1 == qubit2) {                                                                                            \
        return;                                                                                                        \
    }                                                                                                                  \
    ToPermBasis(qubit1);                                                                                               \
    ToPermBasis(qubit2);                                                                                               \
    ApplyEitherControlled(                                                                                             \
        controls, controlLen, { qubit1, qubit2 }, anti,                                                                \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->ctrld; }, [&]() { bare; })
#define CTRL_GEN_ARGS &(mappedControls[0]), mappedControls.size(), trnsMtrx, shards[target].mapped
#define CTRL_ARGS &(mappedControls[0]), mappedControls.size(), mtrx, shards[target].mapped
#define CTRL_1_ARGS mappedControls[0], shards[target].mapped
#define CTRL_2_ARGS mappedControls[0], mappedControls[1], shards[target].mapped
#define CTRL_S_ARGS &(mappedControls[0]), mappedControls.size(), shards[qubit1].mapped, shards[qubit2].mapped
#define CTRL_P_ARGS &(mappedControls[0]), mappedControls.size(), topLeft, bottomRight, shards[target].mapped
#define CTRL_I_ARGS &(mappedControls[0]), mappedControls.size(), topRight, bottomLeft, shards[target].mapped

void QUnit::CNOT(bitLenInt control, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    if (CACHED_PLUS(tShard)) {
        return;
    }

    QEngineShard& cShard = shards[control];
    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (IS_AMP_0(cShard.amp1)) {
            Flush0Eigenstate(control);
            return;
        }
        if (IS_AMP_0(cShard.amp0)) {
            Flush1Eigenstate(control);
            X(target);
            return;
        }
    }

    if (!freezeBasis2Qb) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddInversionAngles(&cShard, ONE_CMPLX, ONE_CMPLX);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    bitLenInt controls[1] = { control };
    const bitLenInt controlLen = 1;

    // We're free to transform gates to any orthonormal basis of the Hilbert space.
    // For a 2 qubit system, if the control is the lefthand bit, it's easy to verify the following truth table for CNOT:
    // |++> -> |++>
    // |+-> -> |-->
    // |-+> -> |-+>
    // |--> -> |+->
    // Under the Jacobian transformation between these two bases for defining the truth table, the matrix representation
    // is equivalent to the gate with bits flipped. We just let ApplyEitherControlled() know to leave the current basis
    // alone, by way of the last optional "true" argument in the call.
    bool pmBasis =
        (cShard.isPauliX && (tShard.isPauliX || tShard.isPauliY) && !QUEUED_PHASE(cShard) && !QUEUED_PHASE(tShard));
    if (pmBasis) {
        RevertBasisY(target);
        std::swap(controls[0], target);
        ApplyEitherControlled(
            controls, controlLen, { target }, false,
            [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->CNOT(CTRL_1_ARGS); },
            [&]() { XBase(target); }, false, true, true);
        return;
    }

    CTRLED_PHASE_INVERT_WRAP(CNOT(CTRL_1_ARGS), MCMtrx(CTRL_GEN_ARGS), X(target), false, true, ONE_CMPLX, ONE_CMPLX);
}

void QUnit::AntiCNOT(bitLenInt control, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    if (CACHED_PLUS(tShard)) {
        return;
    }

    QEngineShard& cShard = shards[control];
    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (IS_AMP_0(cShard.amp1)) {
            Flush0Eigenstate(control);
            X(target);
            return;
        }
        if (IS_AMP_0(cShard.amp0)) {
            Flush1Eigenstate(control);
            return;
        }
    }

    if (!freezeBasis2Qb) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddAntiInversionAngles(&cShard, ONE_CMPLX, ONE_CMPLX);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    const bitLenInt controls[1] = { control };
    const bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(
        AntiCNOT(CTRL_1_ARGS), MACMtrx(CTRL_GEN_ARGS), X(target), true, true, ONE_CMPLX, ONE_CMPLX);
}

void QUnit::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    if (CACHED_PLUS(tShard)) {
        return;
    }

    QEngineShard& c1Shard = shards[control1];
    QEngineShard& c2Shard = shards[control2];

    if (!c1Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c1Shard)) {
            if (IS_AMP_0(c1Shard.amp1)) {
                Flush0Eigenstate(control1);
                return;
            }
            if (IS_AMP_0(c1Shard.amp0)) {
                Flush1Eigenstate(control1);
                CNOT(control2, target);
                return;
            }
        }
    }

    if (!c2Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c2Shard)) {
            if (IS_AMP_0(c2Shard.amp1)) {
                Flush0Eigenstate(control2);
                return;
            }
            if (IS_AMP_0(c2Shard.amp0)) {
                Flush1Eigenstate(control2);
                CNOT(control1, target);
                return;
            }
        }
    }

    if ((!tShard.IsInvertTarget()) && (UNSAFE_CACHED_X(tShard))) {
        H(target);
        CCZ(control1, control2, target);
        H(target);
        return;
    }

    const bitLenInt controls[2] = { control1, control2 };

    ApplyEitherControlled(
        controls, 2, { target }, false,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliX) {
                if (mappedControls.size() == 2) {
                    unit->CCZ(CTRL_2_ARGS);
                } else {
                    unit->CZ(CTRL_1_ARGS);
                }
            } else if (shards[target].isPauliY) {
                if (mappedControls.size() == 2) {
                    unit->CCY(CTRL_2_ARGS);
                } else {
                    unit->CY(CTRL_1_ARGS);
                }
            } else {
                if (mappedControls.size() == 2) {
                    unit->CCNOT(CTRL_2_ARGS);
                } else {
                    unit->CNOT(CTRL_1_ARGS);
                }
            }
        },
        [&]() { X(target); }, false, true);
}

void QUnit::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    if (CACHED_PLUS(tShard)) {
        return;
    }

    const bitLenInt controls[2] = { control1, control2 };

    ApplyEitherControlled(
        controls, 2, { target }, true,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliX) {
                unit->MACPhase(
                    &(mappedControls[0]), mappedControls.size(), ONE_CMPLX, -ONE_CMPLX, shards[target].mapped);
            } else if (shards[target].isPauliY) {
                unit->MACInvert(&(mappedControls[0]), mappedControls.size(), -I_CMPLX, I_CMPLX, shards[target].mapped);
            } else {
                if (mappedControls.size() == 2) {
                    unit->AntiCCNOT(CTRL_2_ARGS);
                } else {
                    unit->AntiCNOT(CTRL_1_ARGS);
                }
            }
        },
        [&]() { X(target); }, false, true);
}

void QUnit::CY(bitLenInt control, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    QEngineShard& cShard = shards[control];

    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (IS_AMP_0(cShard.amp1)) {
            Flush0Eigenstate(control);
            return;
        }
        if (IS_AMP_0(cShard.amp0)) {
            Flush1Eigenstate(control);
            Y(target);
            return;
        }
    }

    if (!freezeBasis2Qb) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddInversionAngles(&cShard, -I_CMPLX, I_CMPLX);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    const bitLenInt controls[1] = { control };
    const bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(CY(CTRL_1_ARGS), MCMtrx(CTRL_GEN_ARGS), Y(target), false, true, -I_CMPLX, I_CMPLX);
}

void QUnit::AntiCY(bitLenInt control, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    QEngineShard& cShard = shards[control];

    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (IS_AMP_0(cShard.amp1)) {
            Flush0Eigenstate(control);
            Y(target);
            return;
        }
        if (IS_AMP_0(cShard.amp0)) {
            Flush1Eigenstate(control);
            return;
        }
    }

    if (!freezeBasis2Qb) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddAntiInversionAngles(&cShard, I_CMPLX, -I_CMPLX);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    const bitLenInt controls[1] = { control };
    const bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(AntiCY(CTRL_1_ARGS), MACMtrx(CTRL_GEN_ARGS), Y(target), true, true, -I_CMPLX, I_CMPLX);
}

void QUnit::CCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    QEngineShard& c1Shard = shards[control1];
    QEngineShard& c2Shard = shards[control2];

    if (!c1Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c1Shard)) {
            if (IS_AMP_0(c1Shard.amp1)) {
                Flush0Eigenstate(control1);
                return;
            }
            if (IS_AMP_0(c1Shard.amp0)) {
                Flush1Eigenstate(control1);
                CY(control2, target);
                return;
            }
        }
    }

    if (!c2Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c2Shard)) {
            if (IS_AMP_0(c2Shard.amp1)) {
                Flush0Eigenstate(control2);
                return;
            }
            if (IS_AMP_0(c2Shard.amp0)) {
                Flush1Eigenstate(control2);
                CY(control1, target);
                return;
            }
        }
    }

    const bitLenInt controls[2] = { control1, control2 };

    ApplyEitherControlled(
        controls, 2, { target }, false,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliX) {
                unit->MCInvert(&(mappedControls[0]), mappedControls.size(), I_CMPLX, -I_CMPLX, shards[target].mapped);
            } else if (shards[target].isPauliY) {
                if (mappedControls.size() == 2) {
                    unit->CCZ(CTRL_2_ARGS);
                } else {
                    unit->CZ(CTRL_1_ARGS);
                }
            } else {
                if (mappedControls.size() == 2) {
                    unit->CCY(CTRL_2_ARGS);
                } else {
                    unit->CY(CTRL_1_ARGS);
                }
            }
        },
        [&]() { Y(target); }, false, true);
}

void QUnit::CZ(bitLenInt control, bitLenInt target)
{
    if ((shards[control].isPauliX || shards[control].isPauliY) && !shards[target].isPauliX &&
        !shards[target].isPauliY) {
        std::swap(control, target);
    }

    QEngineShard& tShard = shards[target];
    QEngineShard& cShard = shards[control];

    if (!tShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(tShard)) {
        if (SHARD_STATE(tShard)) {
            Flush1Eigenstate(target);
            Z(control);
        } else {
            Flush0Eigenstate(target);
        }
        return;
    }

    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (SHARD_STATE(cShard)) {
            Flush1Eigenstate(control);
            Z(target);
        } else {
            Flush0Eigenstate(control);
        }
        return;
    }

    if (!freezeBasis2Qb) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddPhaseAngles(&cShard, ONE_CMPLX, -ONE_CMPLX);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    const bitLenInt controls[1] = { control };
    const bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(CZ(CTRL_1_ARGS), MCMtrx(CTRL_GEN_ARGS), Z(target), false, false, ONE_CMPLX, -ONE_CMPLX);
}

void QUnit::AntiCZ(bitLenInt control, bitLenInt target)
{
    QEngineShard& cShard = shards[control];

    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (SHARD_STATE(cShard)) {
            Flush1Eigenstate(control);
        } else {
            Flush0Eigenstate(control);
            Z(target);
        }
        return;
    }

    QEngineShard& tShard = shards[target];

    if (!freezeBasis2Qb) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddAntiPhaseAngles(&cShard, -ONE_CMPLX, ONE_CMPLX);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    const bitLenInt controls[1] = { control };
    const bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(
        AntiCZ(CTRL_1_ARGS), MACMtrx(CTRL_GEN_ARGS), Z(target), true, false, ONE_CMPLX, -ONE_CMPLX);
}

void QUnit::CH(bitLenInt control, bitLenInt target)
{
    RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);

    QEngineShard& cShard = shards[control];
    if (UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (SHARD_STATE(cShard)) {
            H(target);
        }
        return;
    }

    RevertBasisY(target);
    RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });
    // Should only commute gate with same control and target, if anything:
    CommuteH(target);

    QInterfacePtr unit = Entangle({ control, target });
    unit->CH(shards[control].mapped, shards[target].mapped);

    if (isReactiveSeparate && !freezeTrySeparate && !freezeBasis2Qb) {
        TrySeparate(control);
        TrySeparate(target);
    }
}

void QUnit::AntiCH(bitLenInt control, bitLenInt target)
{
    RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);

    QEngineShard& cShard = shards[control];
    if (UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (!SHARD_STATE(cShard)) {
            H(target);
        }
        return;
    }

    RevertBasisY(target);
    RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });
    // Should only commute gate with same control and target, if anything:
    CommuteH(target);

    QInterfacePtr unit = Entangle({ control, target });
    unit->AntiCH(shards[control].mapped, shards[target].mapped);

    if (isReactiveSeparate && !freezeTrySeparate && !freezeBasis2Qb) {
        TrySeparate(control);
        TrySeparate(target);
    }
}

void QUnit::CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    if ((shards[control1].isPauliX || shards[control1].isPauliY) && !shards[target].isPauliX &&
        !shards[target].isPauliY) {
        std::swap(control1, target);
    }

    if ((shards[control2].isPauliX || shards[control2].isPauliY) && !shards[target].isPauliX &&
        !shards[target].isPauliY) {
        std::swap(control2, target);
    }

    QEngineShard& tShard = shards[target];
    QEngineShard& c1Shard = shards[control1];
    QEngineShard& c2Shard = shards[control2];

    if (!c1Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c1Shard)) {
            if (IS_AMP_0(c1Shard.amp1)) {
                Flush0Eigenstate(control1);
                return;
            }
            if (IS_AMP_0(c1Shard.amp0)) {
                Flush1Eigenstate(control1);
                CZ(control2, target);
                return;
            }
        }
    }

    if (!c2Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c2Shard)) {
            if (IS_AMP_0(c2Shard.amp1)) {
                Flush0Eigenstate(control2);
                return;
            }
            if (IS_AMP_0(c2Shard.amp0)) {
                Flush1Eigenstate(control2);
                CZ(control1, target);
                return;
            }
        }
    }

    if (!tShard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(tShard)) {
            if (IS_AMP_0(tShard.amp1)) {
                Flush0Eigenstate(target);
                return;
            }
            if (IS_AMP_0(tShard.amp0)) {
                Flush1Eigenstate(target);
                CZ(control1, control2);
                return;
            }
        }
    }

    const bitLenInt controls[2] = { control1, control2 };

    ApplyEitherControlled(
        controls, 2, { target }, false,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliX || shards[target].isPauliY) {
                if (mappedControls.size() == 2) {
                    unit->CCNOT(CTRL_2_ARGS);
                } else {
                    unit->CNOT(CTRL_1_ARGS);
                }
            } else {
                if (mappedControls.size() == 2) {
                    unit->CCZ(CTRL_2_ARGS);
                } else {
                    unit->CZ(CTRL_1_ARGS);
                }
            }
        },
        [&]() { Z(target); }, true);
}

void QUnit::Phase(complex topLeft, complex bottomRight, bitLenInt target)
{
    if (randGlobalPhase || IS_1_R1(topLeft)) {
        if (IS_NORM_0(topLeft - bottomRight)) {
            return;
        }

        if (IS_NORM_0(topLeft + bottomRight)) {
            Z(target);
            return;
        }

        if (IS_NORM_0((I_CMPLX * topLeft) - bottomRight)) {
            S(target);
            return;
        }

        if (IS_NORM_0((I_CMPLX * topLeft) + bottomRight)) {
            IS(target);
            return;
        }
    }

    QEngineShard& shard = shards[target];

    shard.CommutePhase(topLeft, bottomRight);

    if (IS_1_R1(topLeft) && UNSAFE_CACHED_ZERO(shard)) {
        Flush0Eigenstate(target);
        return;
    } else if (IS_1_R1(bottomRight) && UNSAFE_CACHED_ONE(shard)) {
        Flush1Eigenstate(target);
        return;
    }

    if (!shard.isPauliX && !shard.isPauliY) {
        if (shard.unit) {
            shard.unit->Phase(topLeft, bottomRight, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.isPhaseDirty = true;
        }

        shard.amp0 *= topLeft;
        shard.amp1 *= bottomRight;
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }

        return;
    }

    complex mtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
    TransformPhase(topLeft, bottomRight, mtrx);

    if (shard.unit) {
        shard.unit->Mtrx(mtrx, shard.mapped);
    }
    if (DIRTY(shard)) {
        shard.MakeDirty();
        return;
    }

    const complex Y0 = shard.amp0;
    shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
    shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
    if (doNormalize) {
        shard.ClampAmps(amplitudeFloor);
    }
}

void QUnit::Invert(complex topRight, complex bottomLeft, bitLenInt target)
{
    if (IS_NORM_0(topRight - bottomLeft) && (randGlobalPhase || IS_1_CMPLX(topRight))) {
        X(target);
        return;
    }

    QEngineShard& shard = shards[target];

    shard.CommutePhase(bottomLeft, topRight);
    shard.FlipPhaseAnti();

    if (shard.isPauliX || shard.isPauliY) {
        complex mtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
        if (shard.isPauliX) {
            TransformXInvert(topRight, bottomLeft, mtrx);
        } else {
            TransformYInvert(topRight, bottomLeft, mtrx);
        }

        if (shard.unit) {
            shard.unit->Mtrx(mtrx, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.MakeDirty();
            return;
        }

        const complex Y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }
    } else {
        if (shard.unit) {
            shard.unit->Invert(topRight, bottomLeft, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.isPhaseDirty = true;
        }

        const complex tempAmp1 = shard.amp0 * bottomLeft;
        shard.amp0 = shard.amp1 * topRight;
        shard.amp1 = tempAmp1;
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }
    }
}

void QUnit::MCPhase(
    const bitLenInt* cControls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    // Commutes with controlled phase optimizations
    if (!controlLen) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if (controlLen == 1) {
        if (IS_NORM_0(topLeft - bottomRight)) {
            Phase(ONE_CMPLX, bottomRight, cControls[0]);
            return;
        }

        if (IS_1_CMPLX(topLeft) && IS_1_CMPLX(-bottomRight)) {
            CZ(cControls[0], target);
            return;
        }
    }

    std::unique_ptr<bitLenInt[]> controls(new bitLenInt[controlLen]);
    std::copy(cControls, cControls + controlLen, controls.get());
    QEngineShard& shard = shards[target];

    if (IS_1_R1(bottomRight) && (!shard.IsInvertTarget() && UNSAFE_CACHED_ONE(shard))) {
        Flush1Eigenstate(target);
        return;
    }

    if (IS_1_R1(topLeft)) {
        if (!shard.IsInvertTarget() && UNSAFE_CACHED_ZERO(shard)) {
            Flush0Eigenstate(target);
            return;
        }

        if (IS_1_R1(-bottomRight)) {
            if (controlLen == 2U) {
                CCZ(controls[0], controls[1], target);
                return;
            }
            if (controlLen == 1U) {
                CZ(controls[0], target);
                return;
            }
        }

        if (!shards[target].isPauliX && !shards[target].isPauliY) {
            for (bitLenInt i = 0; i < controlLen; i++) {
                QEngineShard& cShard = shards[controls[i]];
                if (cShard.isPauliX || cShard.isPauliY) {
                    std::swap(controls[i], target);
                    break;
                }
            }
        }
    }

    if (!freezeBasis2Qb && (controlLen == 1U)) {
        bitLenInt control = controls[0];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];
        if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
            if (SHARD_STATE(cShard)) {
                Flush1Eigenstate(control);
                Phase(topLeft, bottomRight, target);
            } else {
                Flush0Eigenstate(control);
            }

            return;
        }

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_ANTI);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, CTRL_AND_ANTI, {}, { control });

        // This is not a Clifford gate, so we buffer for stabilizer:
        if (!IS_SAME_UNIT(cShard, tShard)) {
            tShard.AddPhaseAngles(&cShard, topLeft, bottomRight);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    ApplyEitherControlled(
        controls.get(), controlLen, { target }, false,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliX || shards[target].isPauliY) {
                complex trnsMtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
                TransformPhase(topLeft, bottomRight, trnsMtrx);
                unit->MCMtrx(CTRL_GEN_ARGS);
            } else {
                unit->MCPhase(CTRL_P_ARGS);
            }
        },
        [&]() { Phase(topLeft, bottomRight, target); }, true);
}

void QUnit::MCInvert(
    const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    if (IS_1_R1(topRight) && IS_1_R1(bottomLeft)) {
        if (controlLen == 2U) {
            CCNOT(controls[0], controls[1], target);
            return;
        }
        if (controlLen == 1U) {
            CNOT(controls[0], target);
            return;
        }
    }

    if (IS_1_R1(I_CMPLX * topRight) && IS_1_R1(-I_CMPLX * bottomLeft)) {
        if (controlLen == 2U) {
            CCY(controls[0], controls[1], target);
            return;
        }
        if (controlLen == 1U) {
            CY(controls[0], target);
            return;
        }
    }

    if (!freezeBasis2Qb && (controlLen == 1U)) {
        bitLenInt control = controls[0];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];
        if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
            if (SHARD_STATE(cShard)) {
                Flush1Eigenstate(control);
                Invert(topRight, bottomLeft, target);
            } else {
                Flush0Eigenstate(control);
            }

            return;
        }

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_PHASE, CONTROLS_AND_TARGETS, ONLY_ANTI);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddInversionAngles(&cShard, topRight, bottomLeft);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(MCInvert(CTRL_I_ARGS), MCMtrx(CTRL_GEN_ARGS), Invert(topRight, bottomLeft, target), false,
        true, topRight, bottomLeft);
}

void QUnit::MACPhase(
    const bitLenInt* cControls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    // Commutes with controlled phase optimizations
    if (!controlLen) {
        Phase(topLeft, bottomRight, target);
        return;
    }

    if (controlLen == 1) {
        if (IS_NORM_0(topLeft - bottomRight)) {
            Phase(topLeft, ONE_CMPLX, cControls[0]);
            return;
        }

        if (IS_1_CMPLX(topLeft) && IS_1_CMPLX(-bottomRight)) {
            AntiCZ(cControls[0], target);
            return;
        }
    }

    std::unique_ptr<bitLenInt[]> controls(new bitLenInt[controlLen]);
    std::copy(cControls, cControls + controlLen, controls.get());
    QEngineShard& shard = shards[target];

    if (IS_1_R1(topLeft) && (!shard.IsInvertTarget() && UNSAFE_CACHED_ZERO(shard))) {
        Flush0Eigenstate(target);
        return;
    }

    if (IS_1_R1(bottomRight)) {
        if (!shard.IsInvertTarget() && UNSAFE_CACHED_ONE(shard)) {
            Flush1Eigenstate(target);
            return;
        }

        if (!shards[target].isPauliX && !shards[target].isPauliY) {
            for (bitLenInt i = 0; i < controlLen; i++) {
                QEngineShard& cShard = shards[controls[i]];
                if (cShard.isPauliX || cShard.isPauliY) {
                    std::swap(controls[i], target);
                    break;
                }
            }
        }
    }

    if (!freezeBasis2Qb && (controlLen == 1U)) {
        bitLenInt control = controls[0];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];
        if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
            if (SHARD_STATE(cShard)) {
                Flush1Eigenstate(control);
            } else {
                Flush0Eigenstate(control);
                Phase(topLeft, bottomRight, target);
            }
            return;
        }

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, ONLY_CTRL);
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, CTRL_AND_ANTI, {}, { control });

        // If this is not a Clifford gate, we buffer for stabilizer:
        if (!IS_SAME_UNIT(cShard, tShard)) {
            tShard.AddAntiPhaseAngles(&cShard, bottomRight, topLeft);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    ApplyEitherControlled(
        controls.get(), controlLen, { target }, true,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliX || shards[target].isPauliY) {
                complex trnsMtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
                TransformPhase(topLeft, bottomRight, trnsMtrx);
                unit->MACMtrx(CTRL_GEN_ARGS);
            } else {
                unit->MACPhase(CTRL_P_ARGS);
            }
        },
        [&]() { Phase(topLeft, bottomRight, target); }, true);
}

void QUnit::MACInvert(
    const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    if (IS_1_R1(topRight) && IS_1_R1(bottomLeft)) {
        if (controlLen == 2U) {
            AntiCCNOT(controls[0], controls[1], target);
            return;
        }
        if (controlLen == 1U) {
            AntiCNOT(controls[0], target);
            return;
        }
    }

    if (IS_1_R1(I_CMPLX * topRight) && IS_1_R1(-I_CMPLX * bottomLeft)) {
        if (controlLen == 1U) {
            AntiCY(controls[0], target);
            return;
        }
    }

    if (!freezeBasis2Qb && (controlLen == 1U)) {
        bitLenInt control = controls[0];
        QEngineShard& cShard = shards[control];
        QEngineShard& tShard = shards[target];
        if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
            if (SHARD_STATE(cShard)) {
                Flush1Eigenstate(control);
            } else {
                Flush0Eigenstate(control);
                Invert(topRight, bottomLeft, target);
            }

            return;
        }

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_PHASE, CONTROLS_AND_TARGETS, ONLY_CTRL);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && (isReactiveSeparate || !ARE_CLIFFORD(cShard, tShard))) {
            tShard.AddAntiInversionAngles(&cShard, bottomLeft, topRight);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(MACInvert(CTRL_I_ARGS), MACMtrx(CTRL_GEN_ARGS), Invert(topRight, bottomLeft, target), true,
        true, topRight, bottomLeft);
}

void QUnit::Mtrx(const complex* mtrx, bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        Phase(mtrx[0], mtrx[3], target);
        return;
    }
    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        Invert(mtrx[1], mtrx[2], target);
        return;
    }
    if ((randGlobalPhase || IS_SAME(mtrx[0], complex(SQRT1_2_R1, ZERO_R1))) && IS_SAME(mtrx[0], mtrx[1]) &&
        IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        H(target);
        return;
    }
    if ((randGlobalPhase || IS_SAME(mtrx[0], complex(SQRT1_2_R1, ZERO_R1))) && IS_SAME(mtrx[0], mtrx[1]) &&
        IS_SAME(mtrx[0], -I_CMPLX * mtrx[2]) && IS_SAME(mtrx[0], I_CMPLX * mtrx[3])) {
        H(target);
        S(target);
        return;
    }
    if ((randGlobalPhase || IS_SAME(mtrx[0], complex(SQRT1_2_R1, ZERO_R1))) && IS_SAME(mtrx[0], I_CMPLX * mtrx[1]) &&
        IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -I_CMPLX * mtrx[3])) {
        IS(target);
        H(target);
        return;
    }

    RevertBasis2Qb(target);

    complex trnsMtrx[4];

    if (shard.isPauliY) {
        TransformY2x2(mtrx, trnsMtrx);
    } else if (shard.isPauliX) {
        TransformX2x2(mtrx, trnsMtrx);
    } else {
        std::copy(mtrx, mtrx + 4, trnsMtrx);
    }

    if (IS_NORM_0(trnsMtrx[1]) && IS_NORM_0(trnsMtrx[2])) {
        const bool wasPauliX = shard.isPauliX;
        const bool wasPauliY = shard.isPauliY;
        shard.isPauliX = false;
        shard.isPauliY = false;
        Phase(trnsMtrx[0], trnsMtrx[3], target);
        shard.isPauliX = wasPauliX;
        shard.isPauliY = wasPauliY;
        return;
    }

    if (IS_NORM_0(trnsMtrx[0]) && IS_NORM_0(trnsMtrx[3])) {
        const bool wasPauliX = shard.isPauliX;
        const bool wasPauliY = shard.isPauliY;
        shard.isPauliX = false;
        shard.isPauliY = false;
        Invert(trnsMtrx[1], trnsMtrx[2], target);
        shard.isPauliX = wasPauliX;
        shard.isPauliY = wasPauliY;
        return;
    }

    if (shard.unit) {
        shard.unit->Mtrx(trnsMtrx, shard.mapped);
    }
    if (DIRTY(shard)) {
        shard.MakeDirty();
        return;
    }

    const complex Y0 = shard.amp0;
    shard.amp0 = (trnsMtrx[0] * Y0) + (trnsMtrx[1] * shard.amp1);
    shard.amp1 = (trnsMtrx[2] * Y0) + (trnsMtrx[3] * shard.amp1);
    if (doNormalize) {
        shard.ClampAmps(amplitudeFloor);
    }
}

void QUnit::MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (IsIdentity(mtrx, true)) {
        return;
    }

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MCPhase(controls, controlLen, mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MCInvert(controls, controlLen, mtrx[1], mtrx[2], target);
        return;
    }

    if ((controlLen == 1U) && (randGlobalPhase || IS_SAME(mtrx[0], (complex)SQRT1_2_R1)) && IS_SAME(mtrx[0], mtrx[1]) &&
        IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        CH(controls[0], target);
        return;
    }

    CTRLED_GEN_WRAP(MCMtrx(CTRL_GEN_ARGS), Mtrx(mtrx, target), false);
}

void QUnit::MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (IsIdentity(mtrx, true)) {
        return;
    }

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MACPhase(controls, controlLen, mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MACInvert(controls, controlLen, mtrx[1], mtrx[2], target);
        return;
    }

    if ((controlLen == 1U) && (randGlobalPhase || IS_SAME(mtrx[0], (complex)SQRT1_2_R1)) && IS_SAME(mtrx[0], mtrx[1]) &&
        IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        AntiCH(controls[0], target);
        return;
    }

    CTRLED_GEN_WRAP(MACMtrx(CTRL_GEN_ARGS), Mtrx(mtrx, target), true);
}

void QUnit::CSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), false);
}

void QUnit::AntiCSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), true);
}

void QUnit::CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), true);
}

void QUnit::CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(CISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    CTRLED_SWAP_WRAP(AntiCISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), true);
}

template <typename CF, typename F>
void QUnit::ApplyEitherControlled(const bitLenInt* controls, bitLenInt controlLen,
    const std::vector<bitLenInt>& targets, bool anti, CF cfn, F fn, bool isPhase, bool isInvert, bool inCurrentBasis)
{
    // If the controls start entirely separated from the targets, it's probably worth checking to see if the have
    // total or no probability of altering the targets, such that we can still keep them separate.

    std::vector<bitLenInt> controlVec;

    for (bitLenInt i = 0; i < controlLen; i++) {
        if (!inCurrentBasis) {
            RevertBasis1Qb(controls[i]);
            RevertBasis2Qb(controls[i], ONLY_INVERT, ONLY_TARGETS);
        }
        QEngineShard& shard = shards[controls[i]];
        if (shard.isProbDirty && shard.isClifford()) {
            ProbBase(controls[i]);
        }
        // If the shard's probability is cached, then it's free to check it, so we advance the loop.
        bool isEigenstate = false;
        if (!shard.isProbDirty) {
            // This might determine that we can just skip out of the whole gate, in which case we return.
            if (IS_AMP_0(shard.amp1)) {
                if (!inCurrentBasis) {
                    Flush0Eigenstate(controls[i]);
                }
                if (!anti) {
                    /* This gate does nothing, so return without applying anything. */
                    return;
                }
                /* This control has 100% chance to "fire," so don't entangle it. */
                isEigenstate = true;
            } else if (IS_AMP_0(shard.amp0)) {
                if (!inCurrentBasis) {
                    Flush1Eigenstate(controls[i]);
                }
                if (anti) {
                    /* This gate does nothing, so return without applying anything. */
                    return;
                }
                /* This control has 100% chance to "fire," so don't entangle it. */
                isEigenstate = true;
            }
        }

        if (!isEigenstate) {
            controlVec.push_back(controls[i]);
        }
    }

    if (!controlVec.size()) {
        // Here, the gate is guaranteed to act as if it wasn't controlled, so we apply the gate without controls,
        // avoiding an entangled representation.
        fn();

        return;
    }

    for (bitLenInt i = 0; i < (bitLenInt)targets.size(); i++) {
        if (isPhase) {
            RevertBasis2Qb(targets[i], ONLY_INVERT, ONLY_TARGETS);
        } else {
            RevertBasis2Qb(targets[i]);
        }
    }

    // TODO: If controls that survive the "first order" check above start out entangled,
    // then it might be worth checking whether there is any intra-unit chance of control bits
    // being conditionally all 0 or all 1, in any unit, due to entanglement.

    // If we've made it this far, we have to form the entangled representation and apply the gate.
    std::vector<bitLenInt> allBits(controlVec.size() + targets.size());
    std::copy(controlVec.begin(), controlVec.end(), allBits.begin());
    std::copy(targets.begin(), targets.end(), allBits.begin() + controlVec.size());
    std::sort(allBits.begin(), allBits.end());
    std::vector<bitLenInt> allBitsMapped(allBits);

    std::vector<bitLenInt*> ebits(allBitsMapped.size());
    for (bitLenInt i = 0; i < (bitLenInt)allBitsMapped.size(); i++) {
        ebits[i] = &allBitsMapped[i];
    }

    QInterfacePtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());

    for (bitLenInt i = 0; i < (bitLenInt)controlVec.size(); i++) {
        shards[controlVec[i]].isPhaseDirty = true;
        controlVec[i] = shards[controlVec[i]].mapped;
    }
    for (bitLenInt i = 0; i < (bitLenInt)targets.size(); i++) {
        QEngineShard& shard = shards[targets[i]];
        shard.isProbDirty |= !isPhase || shard.isPauliX || shard.isPauliY;
        shard.isPhaseDirty = true;
    }

    // This is the original method with the maximum number of non-entangled controls excised, (potentially leaving a
    // target bit in X or Y basis and acting as if Z basis by commutation).
    cfn(unit, controlVec);

    if (!isReactiveSeparate || freezeTrySeparate || freezeBasis2Qb) {
        return;
    }

    // Skip 2-qubit-at-once check for 2 total qubits.
    if (allBits.size() == 2U) {
        TrySeparate(allBits[0]);
        TrySeparate(allBits[1]);
        return;
    }

    // Otherwise, we can try all 2-qubit combinations.
    for (bitLenInt i = 0; i < (bitLenInt)(allBits.size() - 1U); i++) {
        for (bitLenInt j = i + 1; j < (bitLenInt)allBits.size(); j++) {
            TrySeparate(allBits[i], allBits[j]);
        }
    }
}

#if ENABLE_ALU
bool QUnit::CArithmeticOptimize(const bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>* controlVec)
{
    if (!controlLen) {
        return false;
    }

    for (bitLenInt i = 0; i < controlLen; i++) {
        // If any control has a cached zero probability, this gate will do nothing, and we can avoid basically all
        // overhead.
        if (CACHED_ZERO(shards[controls[i]])) {
            return true;
        }
    }

    controlVec->resize(controlLen);
    std::copy(controls, controls + controlLen, controlVec->begin());
    bitLenInt controlIndex = 0;

    for (bitLenInt i = 0; i < controlLen; i++) {
        real1_f prob = Prob(controls[i]);
        if (IS_0_R1(prob)) {
            // If any control has zero probability, this gate will do nothing.
            return true;
        } else if (IS_1_R1(prob)) {
            // If any control has full probability, we can avoid entangling it.
            controlVec->erase(controlVec->begin() + controlIndex);
        } else {
            controlIndex++;
        }
    }

    return false;
}

void QUnit::CINC(bitCapInt toMod, bitLenInt start, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (CArithmeticOptimize(controls, controlLen, &controlVec)) {
        // We've determined we can skip the entire gate.
        return;
    }

    // All cached classical control bits have been removed from controlVec.
    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlVec.size()]);
    std::copy(controlVec.begin(), controlVec.end(), lControls.get());
    DirtyShardIndexVector(controlVec);

    INT(toMod, start, length, 0xFF, false, lControls.get(), controlVec.size());
}

void QUnit::INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    DirtyShardRange(start, length);
    DirtyShardRangePhase(start, length);
    shards[flagIndex].MakeDirty();

    EntangleRange(start, length);
    QInterfacePtr unit = Entangle({ start, flagIndex });
    ((*unit).*fn)(toMod, shards[start].mapped, length, shards[flagIndex].mapped);
}

void QUnit::INCxx(
    INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index)
{
    /* Make sure the flag bits are entangled in the same QU. */
    DirtyShardRange(start, length);
    DirtyShardRangePhase(start, length);
    shards[flag1Index].MakeDirty();
    shards[flag2Index].MakeDirty();

    EntangleRange(start, length);
    QInterfacePtr unit = Entangle({ start, flag1Index, flag2Index });

    ((*unit).*fn)(toMod, shards[start].mapped, length, shards[flag1Index].mapped, shards[flag2Index].mapped);
}

/// Check if overflow arithmetic can be optimized
bool QUnit::INTSOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt overflowIndex)
{
    return INTSCOptimize(toMod, start, length, isAdd, 0xFF, overflowIndex);
}

/// Check if carry arithmetic can be optimized
bool QUnit::INTCOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex)
{
    return INTSCOptimize(toMod, start, length, isAdd, carryIndex, 0xFF);
}

/// Check if arithmetic with both carry and overflow can be optimized
bool QUnit::INTSCOptimize(
    bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex, bitLenInt overflowIndex)
{
    if (!CheckBitsPermutation(start, length)) {
        return false;
    }

    const bool carry = (carryIndex < 0xFF);
    const bool carryIn = carry && M(carryIndex);
    if (carry && (carryIn == isAdd)) {
        toMod++;
    }

    const bitCapInt lengthPower = pow2(length);
    const bitCapInt signMask = pow2(length - 1U);
    const bitCapInt inOutInt = GetCachedPermutation(start, length);
    const bitCapInt inInt = toMod;

    bool isOverflow;
    bitCapInt outInt;
    if (isAdd) {
        isOverflow = (overflowIndex < 0xFF) && isOverflowAdd(inOutInt, inInt, signMask, lengthPower);
        outInt = inOutInt + toMod;
    } else {
        isOverflow = (overflowIndex < 0xFF) && isOverflowSub(inOutInt, inInt, signMask, lengthPower);
        outInt = (inOutInt + lengthPower) - toMod;
    }

    bool carryOut = (outInt >= lengthPower);
    if (carryOut) {
        outInt &= (lengthPower - ONE_BCI);
    }
    if (carry && (carryIn != carryOut)) {
        X(carryIndex);
    }

    SetReg(start, length, outInt);

    if (isOverflow) {
        Z(overflowIndex);
    }

    return true;
}

void QUnit::INT(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex, bool hasCarry,
    const bitLenInt* controls, bitLenInt controlLen)
{
    // Keep the bits separate, if cheap to do so:
    toMod &= pow2Mask(length);
    if (!toMod) {
        return;
    }

    if (!hasCarry && CheckBitsPlus(start, length)) {
        // This operation happens to do nothing.
        return;
    }

    std::vector<bitLenInt> allBits(controlLen + 1U);
    std::copy(controls, controls + controlLen, allBits.begin());
    std::sort(allBits.begin(), allBits.begin() + controlLen);

    std::vector<bitLenInt*> ebits(allBits.size());
    for (bitLenInt i = 0; i < (bitLenInt)(ebits.size() - 1U); i++) {
        ebits[i] = &allBits[i];
    }

    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen]);

    // Try ripple addition, to avoid entanglement.
    const bitLenInt origLength = length;
    bool carry = false;
    bitLenInt i = 0;
    while (i < origLength) {
        bool toAdd = (toMod & ONE_BCI) != 0;

        if (toAdd == carry) {
            toMod >>= ONE_BCI;
            start++;
            length--;
            i++;
            // Nothing is changed, in this bit. (The carry gets promoted to the next bit.)
            continue;
        }

        if (CheckBitPermutation(start)) {
            const bool inReg = SHARD_STATE(shards[start]);
            int total = (toAdd ? 1 : 0) + (inReg ? 1 : 0) + (carry ? 1 : 0);
            if (inReg != (total & 1)) {
                if (controlLen == 1U) {
                    CNOT(controls[0], start);
                } else if (controlLen) {
                    MCInvert(controls, controlLen, ONE_CMPLX, ONE_CMPLX, start);
                } else {
                    X(start);
                }
            }
            carry = (total > 1);

            toMod >>= ONE_BCI;
            start++;
            length--;
            i++;
        } else {
            // The carry-in is classical.
            if (carry) {
                carry = false;
                toMod++;
            }

            if (length == 1) {
                // We need at least two quantum bits left to try to achieve further separability.
                break;
            }

            // We're blocked by needing to add 1 to a bit in an indefinite state, which would superpose the
            // carry-out. However, if we hit another index where the qubit is known and toAdd == inReg, the
            // carry-out is guaranteed not to be superposed.

            // Load the first bit:
            bitCapInt bitMask = ONE_BCI;
            bitCapInt partMod = toMod & bitMask;
            bitLenInt partLength = 1U;
            bitLenInt partStart;
            i++;

            do {
                // Guaranteed to need to load the second bit
                partLength++;
                i++;
                bitMask <<= ONE_BCI;

                toAdd = (toMod & bitMask) != 0;
                partMod |= toMod & bitMask;

                partStart = start + partLength - ONE_BCI;
                if (!CheckBitPermutation(partStart)) {
                    // If the quantum bit at this position is superposed, then we can't determine that the carry
                    // won't be superposed. Advance the loop.
                    continue;
                }

                const bool inReg = SHARD_STATE(shards[partStart]);
                if (toAdd != inReg) {
                    // If toAdd != inReg, the carry out might be superposed. Advance the loop.
                    continue;
                }

                // If toAdd == inReg, this prevents superposition of the carry-out. The carry out of the truth table
                // is independent of the superposed output value of the quantum bit.
                DirtyShardRange(start, partLength);
                EntangleRange(start, partLength);
                if (controlLen) {
                    allBits[controlLen] = start;
                    ebits[controlLen] = &allBits[controlLen];
                    DirtyShardIndexVector(allBits);
                    QInterfacePtr unit = Entangle(ebits);
                    for (bitLenInt cIndex = 0; cIndex < controlLen; cIndex++) {
                        lControls[cIndex] = shards[cIndex].mapped;
                    }
                    unit->CINC(partMod, shards[start].mapped, partLength, lControls.get(), controlLen);
                } else {
                    shards[start].unit->INC(partMod, shards[start].mapped, partLength);
                }

                carry = toAdd;
                toMod >>= (bitCapIntOcl)partLength;
                start += partLength;
                length -= partLength;

                // Break out of the inner loop and return to the flow of the containing loop.
                // (Otherwise, we hit the "continue" calls above.)
                break;
            } while (i < origLength);
        }
    }

    if (!toMod && !length) {
        // We were able to avoid entangling the carry.
        if (hasCarry && carry) {
            if (controlLen == 1U) {
                CNOT(controls[0], carryIndex);
            } else if (controlLen) {
                MCInvert(controls, controlLen, ONE_CMPLX, ONE_CMPLX, carryIndex);
            } else {
                X(carryIndex);
            }
        }
        return;
    }

    // Otherwise, we have one unit left that needs to be entangled, plus carry bit.
    if (hasCarry) {
        if (controlLen) {
            // NOTE: This case is not actually exposed by the public API. It would only become exposed if
            // "CINCC"/"CDECC" were implemented in the public interface, in which case it would become "trivial" to
            // implement, once the QEngine methods were in place.
            throw "ERROR: Controlled-with-carry arithmetic is not implemented!";
        } else {
            DirtyShardRange(start, length);
            shards[carryIndex].MakeDirty();
            EntangleRange(start, length, carryIndex, 1);
            shards[start].unit->INCC(toMod, shards[start].mapped, length, shards[carryIndex].mapped);
        }
    } else {
        DirtyShardRange(start, length);
        EntangleRange(start, length);
        if (controlLen) {
            allBits[controlLen] = start;
            ebits[controlLen] = &allBits[controlLen];
            QInterfacePtr unit = Entangle(ebits);
            DirtyShardIndexVector(allBits);
            for (bitLenInt cIndex = 0; cIndex < controlLen; cIndex++) {
                lControls[cIndex] = shards[cIndex].mapped;
            }
            unit->CINC(toMod, shards[start].mapped, length, lControls.get(), controlLen);
        } else {
            shards[start].unit->INC(toMod, shards[start].mapped, length);
        }
    }
}

void QUnit::INC(bitCapInt toMod, bitLenInt start, bitLenInt length) { INT(toMod, start, length, 0xFF, false); }

/// Add integer (without sign, with carry)
void QUnit::INCC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
        toAdd++;
    }

    INT(toAdd, inOutStart, length, carryIndex, true);
}

/// Subtract integer (without sign, with carry)
void QUnit::DECC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INT(invToSub, inOutStart, length, carryIndex, true);
}

void QUnit::INTS(
    bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex, bool hasCarry)
{
    toMod &= pow2Mask(length);
    if (!toMod) {
        return;
    }

    const bitLenInt signBit = start + length - 1U;
    const bool knewFlagSet = CheckBitPermutation(overflowIndex);
    const bool flagSet = SHARD_STATE(shards[overflowIndex]);

    if (knewFlagSet && !flagSet) {
        // Overflow detection is disabled
        INT(toMod, start, length, carryIndex, hasCarry);
        return;
    }

    const bool addendNeg = (toMod & pow2(length - 1U)) != 0;
    const bool knewSign = CheckBitPermutation(signBit);
    const bool quantumNeg = SHARD_STATE(shards[signBit]);

    if (knewSign && (addendNeg != quantumNeg)) {
        // No chance of overflow
        INT(toMod, start, length, carryIndex, hasCarry);
        return;
    }

    // Otherwise, form the potentially entangled representation:
    if (hasCarry) {
        // Keep the bits separate, if cheap to do so:
        if (INTSCOptimize(toMod, start, length, true, carryIndex, overflowIndex)) {
            return;
        }
        INCxx(&QInterface::INCSC, toMod, start, length, overflowIndex, carryIndex);
    } else {
        // Keep the bits separate, if cheap to do so:
        if (INTSOptimize(toMod, start, length, true, overflowIndex)) {
            return;
        }
        INCx(&QInterface::INCS, toMod, start, length, overflowIndex);
    }
}

void QUnit::INCS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INTS(toMod, start, length, overflowIndex, 0xFF, false);
}

void QUnit::INCSC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
        toAdd++;
    }

    INTS(toAdd, inOutStart, length, overflowIndex, carryIndex, true);
}

void QUnit::DECSC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INTS(invToSub, inOutStart, length, overflowIndex, carryIndex, true);
}

void QUnit::INCSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // The phase effect of the overflow is undetectable, if this check passes:
    if (INTCOptimize(toMod, start, length, true, carryIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCx(&QInterface::INCSC, toMod, start, length, carryIndex);
}

void QUnit::DECSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // The phase effect of the overflow is undetectable, if this check passes:
    if (INTCOptimize(toMod, start, length, false, carryIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCx(&QInterface::DECSC, toMod, start, length, carryIndex);
}

#if ENABLE_BCD
void QUnit::INCBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    // BCD variants are low priority for optimization, for the time being.
    DirtyShardRange(start, length);
    EntangleRange(start, length);
    shards[start].unit->INCBCD(toMod, shards[start].mapped, length);
}

void QUnit::INCBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // BCD variants are low priority for optimization, for the time being.
    INCx(&QInterface::INCBCDC, toMod, start, length, carryIndex);
}

void QUnit::DECBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    // BCD variants are low priority for optimization, for the time being.
    DirtyShardRange(start, length);
    EntangleRange(start, length);
    shards[start].unit->DECBCD(toMod, shards[start].mapped, length);
}

void QUnit::DECBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // BCD variants are low priority for optimization, for the time being.
    INCx(&QInterface::DECBCDC, toMod, start, length, carryIndex);
}
#endif

void QUnit::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (!toMul) {
        SetReg(inOutStart, length, 0U);
        SetReg(carryStart, length, 0U);
        return;
    } else if (toMul == ONE_BCI) {
        SetReg(carryStart, length, 0U);
        return;
    }

    if (CheckBitsPermutation(inOutStart, length)) {
        const bitCapInt lengthMask = pow2Mask(length);
        const bitCapInt res = GetCachedPermutation(inOutStart, length) * toMul;
        SetReg(inOutStart, length, res & lengthMask);
        SetReg(carryStart, length, (res >> (bitCapIntOcl)length) & lengthMask);
        return;
    }

    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->MUL(toMul, shards[inOutStart].mapped, shards[carryStart].mapped, length);
}

void QUnit::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (toDiv == ONE_BCI) {
        return;
    }

    if (CheckBitsPermutation(inOutStart, length) && CheckBitsPermutation(carryStart, length)) {
        const bitCapInt lengthMask = pow2Mask(length);
        const bitCapInt origRes =
            GetCachedPermutation(inOutStart, length) | (GetCachedPermutation(carryStart, length) << length);
        const bitCapInt res = origRes / toDiv;
        if (origRes == (res * toDiv)) {
            SetReg(inOutStart, length, res & lengthMask);
            SetReg(carryStart, length, (res >> (bitCapIntOcl)length) & lengthMask);
        }
        return;
    }

    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->DIV(toDiv, shards[inOutStart].mapped, shards[carryStart].mapped, length);
}

void QUnit::xMULModNOut(
    bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length, bool inverse)
{
    // Inexpensive edge case
    if (!toMod) {
        SetReg(outStart, length, 0U);
        return;
    }

    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(inStart, length)) {
        bitCapInt res = (GetCachedPermutation(inStart, length) * toMod) % modN;
        if (inverse) {
            DEC(res, outStart, length);
        } else {
            SetReg(outStart, length, res);
        }
        return;
    }

    if (!inverse) {
        SetReg(outStart, length, 0U);
    }

    // If "modN" is a power of 2, we have an optimized way of handling this.
    if (isPowerOfTwo(modN)) {
        bool isFullyEntangled = true;
        for (bitLenInt i = 1; i < length; i++) {
            if (shards[inStart].unit != shards[inStart + i].unit) {
                isFullyEntangled = false;
                break;
            }
        }

        if (!isFullyEntangled) {
            bitCapInt toModExp = toMod;
            bitLenInt controls[1];
            for (bitLenInt i = 0; i < length; i++) {
                controls[0] = inStart + i;
                if (inverse) {
                    CDEC(toModExp, outStart, length, controls, 1U);
                } else {
                    CINC(toModExp, outStart, length, controls, 1U);
                }
                toModExp <<= ONE_BCI;
            }
            return;
        }
    }

    DirtyShardRangePhase(inStart, length);
    DirtyShardRange(outStart, length);

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inStart, length, outStart, length);
    if (inverse) {
        shards[inStart].unit->IMULModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    } else {
        shards[inStart].unit->MULModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    }
}

void QUnit::MULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    xMULModNOut(toMod, modN, inStart, outStart, length, false);
}

void QUnit::IMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    xMULModNOut(toMod, modN, inStart, outStart, length, true);
}

void QUnit::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == ONE_BCI) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(inStart, length)) {
        const bitCapInt res = intPow(toMod, GetCachedPermutation(inStart, length)) % modN;
        SetReg(outStart, length, res);
        return;
    }

    SetReg(outStart, length, 0);

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inStart, length, outStart, length);
    shards[inStart].unit->POWModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    DirtyShardRangePhase(inStart, length);
    DirtyShardRange(outStart, length);
}

QInterfacePtr QUnit::CMULEntangle(std::vector<bitLenInt> controlVec, bitLenInt start, bitLenInt carryStart,
    bitLenInt length, std::vector<bitLenInt>* controlsMapped)
{
    DirtyShardRangePhase(start, length);
    DirtyShardRange(carryStart, length);
    EntangleRange(start, length);
    EntangleRange(carryStart, length);

    std::vector<bitLenInt> bits(controlVec.size() + 2);
    for (bitLenInt i = 0; i < (bitLenInt)controlVec.size(); i++) {
        bits[i] = controlVec[i];
    }
    bits[controlVec.size()] = start;
    bits[controlVec.size() + 1] = carryStart;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(bits.size());
    for (bitLenInt i = 0; i < (bitLenInt)ebits.size(); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    if (controlVec.size()) {
        controlsMapped->resize(controlVec.size());
        for (bitLenInt i = 0; i < (bitLenInt)controlVec.size(); i++) {
            (*controlsMapped)[i] = shards[controlVec[i]].mapped;
            shards[controlVec[i]].isPhaseDirty = true;
        }
    }

    return unit;
}

void QUnit::CMULx(CMULFn fn, bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (CArithmeticOptimize(controls, controlLen, &controlVec)) {
        // We've determined we can skip the entire operation:
        return;
    }

    // Otherwise, we have to "dirty" the register.
    std::vector<bitLenInt> controlsMapped;
    QInterfacePtr unit = CMULEntangle(controlVec, start, carryStart, length, &controlsMapped);

    ((*unit).*fn)(toMod, shards[start].mapped, shards[carryStart].mapped, length,
        controlVec.size() ? &(controlsMapped[0]) : NULL, controlVec.size());

    DirtyShardRange(start, length);
}

void QUnit::CMULModx(CMULModFn fn, bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
    bitLenInt length, std::vector<bitLenInt> controlVec)
{
    std::vector<bitLenInt> controlsMapped;
    QInterfacePtr unit = CMULEntangle(controlVec, start, carryStart, length, &controlsMapped);

    ((*unit).*fn)(toMod, modN, shards[start].mapped, shards[carryStart].mapped, length,
        controlVec.size() ? &(controlsMapped[0]) : NULL, controlVec.size());

    DirtyShardRangePhase(start, length);
}

void QUnit::CMUL(bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
    bitLenInt controlLen)
{
    if (!controlLen) {
        MUL(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QInterface::CMUL, toMod, start, carryStart, length, controls, controlLen);
}

void QUnit::CDIV(bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
    bitLenInt controlLen)
{
    if (!controlLen) {
        DIV(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QInterface::CDIV, toMod, start, carryStart, length, controls, controlLen);
}

void QUnit::CxMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen, bool inverse)
{
    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (CArithmeticOptimize(controls, controlLen, &controlVec)) {
        // We've determined we can skip the entire operation:
        return;
    }

    if (!controlVec.size()) {
        if (inverse) {
            IMULModNOut(toMod, modN, inStart, outStart, length);
        } else {
            MULModNOut(toMod, modN, inStart, outStart, length);
        }
        return;
    }

    if (!inverse) {
        SetReg(outStart, length, 0U);
    }

    // If "modN" is a power of 2, we have an optimized way of handling this.
    if (isPowerOfTwo(modN)) {
        bool isFullyEntangled = true;
        for (bitLenInt i = 1; i < length; i++) {
            if (shards[inStart].unit != shards[inStart + i].unit) {
                isFullyEntangled = false;
                break;
            }
        }

        if (!isFullyEntangled) {
            bitCapInt toModExp = toMod;
            std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlVec.size() + 1U]);
            std::copy(controlVec.begin(), controlVec.end(), lControls.get());
            for (bitLenInt i = 0; i < length; i++) {
                lControls[controlVec.size()] = inStart + i;
                if (inverse) {
                    CDEC(toModExp, outStart, length, lControls.get(), controlVec.size() + 1U);
                } else {
                    CINC(toModExp, outStart, length, lControls.get(), controlVec.size() + 1U);
                }
                toModExp <<= ONE_BCI;
            }
            return;
        }
    }

    if (inverse) {
        CMULModx(&QInterface::CIMULModNOut, toMod, modN, inStart, outStart, length, controlVec);
    } else {
        CMULModx(&QInterface::CMULModNOut, toMod, modN, inStart, outStart, length, controlVec);
    }
}

void QUnit::CMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CxMULModNOut(toMod, modN, inStart, outStart, length, controls, controlLen, false);
}

void QUnit::CIMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CxMULModNOut(toMod, modN, inStart, outStart, length, controls, controlLen, true);
}

void QUnit::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0U);

    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (CArithmeticOptimize(controls, controlLen, &controlVec)) {
        // We've determined we can skip the entire operation:
        return;
    }

    CMULModx(&QInterface::CPOWModNOut, toMod, modN, inStart, outStart, length, controlVec);
}

bitCapInt QUnit::GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values)
{
    const bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(indexStart, indexLength);
    const bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt value = 0;
    for (bitCapIntOcl j = 0; j < valueBytes; j++) {
        value |= (bitCapInt)values[indexInt * valueBytes + j] << (8U * j);
    }

    return value;
}

bitCapInt QUnit::GetIndexedEigenstate(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    const bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(start, length);
    const bitLenInt bytes = (length + 7U) / 8U;
    bitCapInt value = 0;
    for (bitCapIntOcl j = 0; j < bytes; j++) {
        value |= (bitCapInt)values[indexInt * bytes + j] << (8U * j);
    }

    return value;
}

bitCapInt QUnit::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    const unsigned char* values, bool resetValue)
{
    // TODO: Index bits that have exactly 0 or 1 probability can be optimized out of the gate.
    // This could follow the logic of UniformlyControlledSingleBit().
    // In the meantime, checking if all index bits are in eigenstates takes very little overhead.
    if (CheckBitsPermutation(indexStart, indexLength)) {
        const bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        SetReg(valueStart, valueLength, value);
#if ENABLE_VM6502Q_DEBUG
        return value;
#else
        return 0;
#endif
    }

    EntangleRange(indexStart, indexLength, valueStart, valueLength);

    const bitCapInt toRet = shards[indexStart].unit->IndexedLDA(
        shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, values, resetValue);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);

    return toRet;
}

bitCapInt QUnit::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, const unsigned char* values)
{
#if ENABLE_VM6502Q_DEBUG
    if (CheckBitsPermutation(indexStart, indexLength) && CheckBitsPermutation(valueStart, valueLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        value = GetCachedPermutation(valueStart, valueLength) + value;
        const bitCapInt valueMask = pow2Mask(valueLength);
        bool carry = false;
        if (value > valueMask) {
            value &= valueMask;
            carry = true;
        }
        SetReg(valueStart, valueLength, value);
        if (carry) {
            X(carryIndex);
        }
        return value;
    }
#else
    if (CheckBitsPermutation(indexStart, indexLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        INCC(value, valueStart, valueLength, carryIndex);
        return 0;
    }
#endif
    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    const bitCapInt toRet = shards[indexStart].unit->IndexedADC(shards[indexStart].mapped, indexLength,
        shards[valueStart].mapped, valueLength, shards[carryIndex].mapped, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].MakeDirty();

    return toRet;
}

bitCapInt QUnit::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, const unsigned char* values)
{
#if ENABLE_VM6502Q_DEBUG
    if (CheckBitsPermutation(indexStart, indexLength) && CheckBitsPermutation(valueStart, valueLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        value = GetCachedPermutation(valueStart, valueLength) - value;
        const bitCapInt valueMask = pow2Mask(valueLength);
        bool carry = false;
        if (value > valueMask) {
            value &= valueMask;
            carry = true;
        }
        SetReg(valueStart, valueLength, value);
        if (carry) {
            X(carryIndex);
        }
        return value;
    }
#else
    if (CheckBitsPermutation(indexStart, indexLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        DECC(value, valueStart, valueLength, carryIndex);
        return 0;
    }
#endif
    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    const bitCapInt toRet = shards[indexStart].unit->IndexedSBC(shards[indexStart].mapped, indexLength,
        shards[valueStart].mapped, valueLength, shards[carryIndex].mapped, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].MakeDirty();

    return toRet;
}

void QUnit::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    if (CheckBitsPlus(start, length)) {
        // This operation happens to do nothing.
        return;
    }

    if (CheckBitsPermutation(start, length)) {
        const bitCapInt value = GetIndexedEigenstate(start, length, values);
        SetReg(start, length, value);
        return;
    }

    DirtyShardRange(start, length);
    EntangleRange(start, length);
    shards[start].unit->Hash(shards[start].mapped, length, values);
}

void QUnit::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(start, length)) {
        if (GetCachedPermutation(start, length) < greaterPerm) {
            // This has no physical effect, but we do it to respect direct simulator check of amplitudes:
            QEngineShard& shard = shards[start];
            if (shard.unit) {
                shard.unit->PhaseFlip();
            }

            shard.amp0 = -shard.amp0;
            shard.amp1 = -shard.amp1;
        }
        return;
    }

    // Otherwise, form the potentially entangled representation:
    DirtyShardRange(start, length);
    EntangleRange(start, length);
    shards[start].unit->PhaseFlipIfLess(greaterPerm, shards[start].mapped, length);
}

void QUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (!shards[flagIndex].isProbDirty) {
        real1_f prob = Prob(flagIndex);
        if (IS_0_R1(prob)) {
            return;
        } else if (IS_1_R1(prob)) {
            PhaseFlipIfLess(greaterPerm, start, length);
            return;
        }
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(start, length, flagIndex, 1);
    shards[start].unit->CPhaseFlipIfLess(greaterPerm, shards[start].mapped, length, shards[flagIndex].mapped);
    DirtyShardRange(start, length);
    shards[flagIndex].isPhaseDirty = true;
}
#endif

bool QUnit::ParallelUnitApply(ParallelUnitFn fn, real1_f param1, real1_f param2, int32_t param3)
{
    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (toFind && (find(units.begin(), units.end(), toFind) == units.end())) {
            units.push_back(toFind);
            if (!fn(toFind, param1, param2, param3)) {
                return false;
            }
        }
    }

    return true;
}

void QUnit::UpdateRunningNorm(real1_f norm_thresh)
{
    ParallelUnitApply(
        [](QInterfacePtr unit, real1_f norm_thresh, real1_f unused2, int32_t unused3) {
            unit->UpdateRunningNorm(norm_thresh);
            return true;
        },
        norm_thresh);
}

void QUnit::NormalizeState(real1_f nrm, real1_f norm_thresh)
{
    ParallelUnitApply(
        [](QInterfacePtr unit, real1_f nrm, real1_f norm_thresh, int32_t unused) {
            unit->NormalizeState(nrm, norm_thresh);
            return true;
        },
        nrm, norm_thresh);
}

void QUnit::Finish()
{
    ParallelUnitApply([](QInterfacePtr unit, real1_f unused1, real1_f unused2, int32_t unused3) {
        unit->Finish();
        return true;
    });
}

bool QUnit::isFinished()
{
    return ParallelUnitApply(
        [](QInterfacePtr unit, real1_f unused1, real1_f unused2, int32_t unused3) { return unit->isFinished(); });
}

void QUnit::SetDevice(int dID, bool forceReInit)
{
    devID = dID;
    ParallelUnitApply(
        [](QInterfacePtr unit, real1_f unused1, real1_f forceReInit, int32_t dID) {
            unit->SetDevice(dID, (forceReInit > 0.5));
            return true;
        },
        ZERO_R1, forceReInit ? ONE_R1 : ZERO_R1, dID);
}

real1_f QUnit::SumSqrDiff(QUnitPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1;
    }

    if (qubitCount == 1U) {
        RevertBasis1Qb(0);
        toCompare->RevertBasis1Qb(0);

        complex mAmps[2], oAmps[2];
        if (shards[0].unit) {
            shards[0].unit->GetQuantumState(mAmps);
        } else {
            mAmps[0] = shards[0].amp0;
            mAmps[1] = shards[0].amp1;
        }
        if (!toCompare->shards[0].unit) {
            toCompare->shards[0].unit->GetQuantumState(oAmps);
        } else {
            oAmps[0] = toCompare->shards[0].amp0;
            oAmps[1] = toCompare->shards[0].amp1;
        }

        return norm(mAmps[0] - oAmps[0]) + norm(mAmps[1] - oAmps[1]);
    }

    if (CheckBitsPermutation(0, qubitCount) && toCompare->CheckBitsPermutation(0, qubitCount)) {
        if (GetCachedPermutation((bitLenInt)0, qubitCount) ==
            toCompare->GetCachedPermutation((bitLenInt)0, qubitCount)) {
            return ZERO_R1;
        }

        // Necessarily max difference:
        return ONE_R1;
    }

    QUnitPtr thisCopyShared, thatCopyShared;
    QUnit* thisCopy;
    QUnit* thatCopy;

    if (shards[0].GetQubitCount() == qubitCount) {
        ToPermBasisAll();
        OrderContiguous(shards[0].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnit>(Clone());
        thisCopyShared->EntangleAll();
        thisCopy = thisCopyShared.get();
    }

    if (toCompare->shards[0].GetQubitCount() == qubitCount) {
        toCompare->ToPermBasisAll();
        toCompare->OrderContiguous(toCompare->shards[0].unit);
        thatCopy = toCompare.get();
    } else {
        thatCopyShared = std::dynamic_pointer_cast<QUnit>(toCompare->Clone());
        thatCopyShared->EntangleAll();
        thatCopy = thatCopyShared.get();
    }

    return thisCopy->shards[0].unit->SumSqrDiff(thatCopy->shards[0].unit);
}

QInterfacePtr QUnit::Clone()
{
    // TODO: Copy buffers instead of flushing?
    for (bitLenInt i = 0; i < qubitCount; i++) {
        RevertBasis2Qb(i);
    }

    QUnitPtr copyPtr = std::make_shared<QUnit>(engines, qubitCount, 0, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits,
        separabilityThreshold);

    Finish();
    copyPtr->Finish();
    copyPtr->SetReactiveSeparate(isReactiveSeparate);

    return CloneBody(copyPtr);
}

QInterfacePtr QUnit::CloneBody(QUnitPtr copyPtr)
{
    std::map<QInterfacePtr, QInterfacePtr> dupeEngines;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        copyPtr->shards[i] = QEngineShard(shards[i]);

        QInterfacePtr unit = shards[i].unit;
        if (!unit) {
            continue;
        }

        if (dupeEngines.find(unit) == dupeEngines.end()) {
            dupeEngines[unit] = unit->Clone();
        }

        copyPtr->shards[i].unit = dupeEngines[unit];
    }

    return copyPtr;
}

void QUnit::ApplyBuffer(PhaseShardPtr phaseShard, bitLenInt control, bitLenInt target, bool isAnti)
{
    const bitLenInt controls[1] = { control };

    const complex polarDiff = phaseShard->cmplxDiff;
    const complex polarSame = phaseShard->cmplxSame;

    freezeBasis2Qb = true;
    if (phaseShard->isInvert) {
        if (isAnti) {
            MACInvert(controls, 1U, polarSame, polarDiff, target);
        } else {
            MCInvert(controls, 1U, polarDiff, polarSame, target);
        }
    } else {
        if (isAnti) {
            MACPhase(controls, 1U, polarSame, polarDiff, target);
        } else {
            MCPhase(controls, 1U, polarDiff, polarSame, target);
        }
    }
    freezeBasis2Qb = false;
}

void QUnit::ApplyBufferMap(bitLenInt bitIndex, ShardToPhaseMap bufferMap, RevertExclusivity exclusivity, bool isControl,
    bool isAnti, const std::set<bitLenInt>& exceptPartners, bool dumpSkipped)
{
    QEngineShard& shard = shards[bitIndex];

    ShardToPhaseMap::iterator phaseShard;

    while (bufferMap.size() > 0) {
        phaseShard = bufferMap.begin();
        QEngineShardPtr partner = phaseShard->first;

        if (((exclusivity == ONLY_INVERT) && !phaseShard->second->isInvert) ||
            ((exclusivity == ONLY_PHASE) && phaseShard->second->isInvert)) {
            bufferMap.erase(phaseShard);
            if (dumpSkipped) {
                shard.RemoveTarget(partner);
            }
            continue;
        }

        bitLenInt partnerIndex = FindShardIndex(partner);

        if (exceptPartners.find(partnerIndex) != exceptPartners.end()) {
            bufferMap.erase(phaseShard);
            if (dumpSkipped) {
                if (isControl) {
                    if (isAnti) {
                        shard.RemoveAntiTarget(partner);
                    } else {
                        shard.RemoveTarget(partner);
                    }
                } else {
                    if (isAnti) {
                        shard.RemoveAntiControl(partner);
                    } else {
                        shard.RemoveControl(partner);
                    }
                }
            }
            continue;
        }

        if (isControl) {
            if (isAnti) {
                shard.RemoveAntiTarget(partner);
            } else {
                shard.RemoveTarget(partner);
            }
            ApplyBuffer(phaseShard->second, bitIndex, partnerIndex, isAnti);
        } else {
            if (isAnti) {
                shard.RemoveAntiControl(partner);
            } else {
                shard.RemoveControl(partner);
            }
            ApplyBuffer(phaseShard->second, partnerIndex, bitIndex, isAnti);
        }

        bufferMap.erase(phaseShard);
    }
}

void QUnit::RevertBasis2Qb(bitLenInt i, RevertExclusivity exclusivity, RevertControl controlExclusivity,
    RevertAnti antiExclusivity, const std::set<bitLenInt>& exceptControlling,
    const std::set<bitLenInt>& exceptTargetedBy, bool dumpSkipped, bool skipOptimize)
{
    QEngineShard& shard = shards[i];

    if (freezeBasis2Qb || !QUEUED_PHASE(shard)) {
        // Recursive call that should be blocked,
        // or already in target basis.
        return;
    }

    shard.CombineGates();

    if (!skipOptimize && (controlExclusivity == ONLY_CONTROLS) && (exclusivity != ONLY_INVERT)) {
        if (antiExclusivity != ONLY_ANTI) {
            shard.OptimizeControls();
        }
        if (antiExclusivity != ONLY_CTRL) {
            shard.OptimizeAntiControls();
        }
    } else if (!skipOptimize && (controlExclusivity == ONLY_TARGETS) && (exclusivity != ONLY_INVERT)) {
        if (antiExclusivity == CTRL_AND_ANTI) {
            shard.OptimizeBothTargets();
        } else if (antiExclusivity == ONLY_CTRL) {
            shard.OptimizeTargets();
        } else if (antiExclusivity == ONLY_ANTI) {
            shard.OptimizeAntiTargets();
        }
    }

    if (controlExclusivity != ONLY_TARGETS) {
        if (antiExclusivity != ONLY_ANTI) {
            ApplyBufferMap(i, shard.controlsShards, exclusivity, true, false, exceptControlling, dumpSkipped);
        }
        if (antiExclusivity != ONLY_CTRL) {
            ApplyBufferMap(i, shard.antiControlsShards, exclusivity, true, true, exceptControlling, dumpSkipped);
        }
    }

    if (controlExclusivity == ONLY_CONTROLS) {
        return;
    }

    if (antiExclusivity != ONLY_ANTI) {
        ApplyBufferMap(i, shard.targetOfShards, exclusivity, false, false, exceptTargetedBy, dumpSkipped);
    }
    if (antiExclusivity != ONLY_CTRL) {
        ApplyBufferMap(i, shard.antiTargetOfShards, exclusivity, false, true, exceptTargetedBy, dumpSkipped);
    }
}

void QUnit::CommuteH(bitLenInt bitIndex)
{
    QEngineShard& shard = shards[bitIndex];

    if (!QUEUED_PHASE(shard)) {
        return;
    }

    ShardToPhaseMap controlsShards = shard.controlsShards;

    for (auto phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
        PhaseShardPtr buffer = phaseShard->second;
        QEngineShardPtr partner = phaseShard->first;

        if (buffer->isInvert) {
            continue;
        }

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        if (IS_ARG_0(polarDiff) && IS_ARG_PI(polarSame)) {
            shard.RemoveTarget(partner);
            shard.AddPhaseAngles(partner, ONE_CMPLX, -ONE_CMPLX);
        } else if (IS_ARG_PI(polarDiff) && IS_ARG_0(polarSame)) {
            shard.RemoveTarget(partner);
            shard.AddAntiPhaseAngles(partner, -ONE_CMPLX, ONE_CMPLX);
        }
    }

    controlsShards = shard.antiControlsShards;

    for (auto phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
        PhaseShardPtr buffer = phaseShard->second;
        QEngineShardPtr partner = phaseShard->first;

        if (buffer->isInvert) {
            continue;
        }

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        if (IS_ARG_0(polarDiff) && IS_ARG_PI(polarSame)) {
            shard.RemoveAntiTarget(partner);
            shard.AddAntiPhaseAngles(partner, ONE_CMPLX, -ONE_CMPLX);
        } else if (IS_ARG_PI(polarDiff) && IS_ARG_0(polarSame)) {
            shard.RemoveAntiTarget(partner);
            shard.AddPhaseAngles(partner, -ONE_CMPLX, ONE_CMPLX);
        }
    }

    RevertBasis2Qb(bitIndex, INVERT_AND_PHASE, ONLY_CONTROLS, CTRL_AND_ANTI, {}, {}, false, true);

    if (!QUEUED_PHASE(shard)) {
        return;
    }

    std::map<QEngineShardPtr, bool> isPhaseMap;
    std::map<QEngineShardPtr, bool> isInvertMap;
    ShardToPhaseMap targetOfShards = shard.targetOfShards;

    for (auto phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
        PhaseShardPtr buffer = phaseShard->second;

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        QEngineShardPtr partner = phaseShard->first;

        const bool isSame = buffer->isInvert && IS_SAME(polarDiff, polarSame);
        if (isSame) {
            isPhaseMap[partner] = true;
            continue;
        }

        const bool isOpposite = IS_OPPOSITE(polarDiff, polarSame);
        if (isOpposite) {
            isInvertMap[partner] = true;
            continue;
        }

        bitLenInt control = FindShardIndex(partner);
        shard.RemoveControl(partner);
        ApplyBuffer(buffer, control, bitIndex, false);
    }

    targetOfShards = shard.antiTargetOfShards;

    for (auto phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
        PhaseShardPtr buffer = phaseShard->second;

        const complex polarDiff = buffer->cmplxDiff;
        const complex polarSame = buffer->cmplxSame;

        QEngineShardPtr partner = phaseShard->first;

        if (!isInvertMap[partner]) {
            const bool isSame = buffer->isInvert && IS_SAME(polarDiff, polarSame);
            if (isSame) {
                continue;
            }

            const bool isOpposite = IS_OPPOSITE(polarDiff, polarSame) && !isPhaseMap[partner];
            if (isOpposite) {
                continue;
            }
        }

        const bitLenInt control = FindShardIndex(partner);
        shard.RemoveAntiControl(partner);
        ApplyBuffer(buffer, control, bitIndex, true);
    }

    shard.CommuteH();
}

void QUnit::OptimizePairBuffers(bitLenInt control, bitLenInt target, bool anti)
{
    QEngineShard& cShard = shards[control];
    QEngineShard& tShard = shards[target];

    ShardToPhaseMap& targets = anti ? tShard.antiTargetOfShards : tShard.targetOfShards;
    ShardToPhaseMap::iterator phaseShard = targets.find(&cShard);
    if (phaseShard == targets.end()) {
        return;
    }

    PhaseShardPtr buffer = phaseShard->second;

    if (!buffer->isInvert) {
        if (anti) {
            if (IS_1_CMPLX(buffer->cmplxDiff) && IS_1_CMPLX(buffer->cmplxSame)) {
                tShard.RemoveAntiControl(&cShard);
                return;
            }
            if (IS_SAME_UNIT(cShard, tShard) || (!isReactiveSeparate && ARE_CLIFFORD(cShard, tShard))) {
                tShard.RemoveAntiControl(&cShard);
                ApplyBuffer(buffer, control, target, true);
                return;
            }
        } else {
            if (IS_1_CMPLX(buffer->cmplxDiff) && IS_1_CMPLX(buffer->cmplxSame)) {
                tShard.RemoveControl(&cShard);
                return;
            }
            if (IS_SAME_UNIT(cShard, tShard) || (!isReactiveSeparate && ARE_CLIFFORD(cShard, tShard))) {
                tShard.RemoveControl(&cShard);
                ApplyBuffer(buffer, control, target, false);
                return;
            }
        }
    }

    ShardToPhaseMap& antiTargets = anti ? tShard.targetOfShards : tShard.antiTargetOfShards;
    ShardToPhaseMap::iterator antiShard = antiTargets.find(&cShard);
    if (antiShard == antiTargets.end()) {
        return;
    }

    PhaseShardPtr aBuffer = antiShard->second;

    if (buffer->isInvert != aBuffer->isInvert) {
        return;
    }

    if (anti) {
        std::swap(buffer, aBuffer);
    }

    const bool isInvert = buffer->isInvert;
    if (isInvert) {
        if (tShard.isPauliY) {
            YBase(target);
        } else if (tShard.isPauliX) {
            ZBase(target);
        } else {
            XBase(target);
        }

        buffer->isInvert = false;
        aBuffer->isInvert = false;
    }

    if (IS_NORM_0(buffer->cmplxDiff - aBuffer->cmplxSame) && IS_NORM_0(buffer->cmplxSame - aBuffer->cmplxDiff)) {
        tShard.RemoveControl(&cShard);
        tShard.RemoveAntiControl(&cShard);
        Phase(buffer->cmplxDiff, buffer->cmplxSame, target);
    } else if (isInvert) {
        if (IS_1_CMPLX(buffer->cmplxDiff) && IS_1_CMPLX(buffer->cmplxSame)) {
            tShard.RemoveControl(&cShard);
        }
        if (IS_1_CMPLX(aBuffer->cmplxDiff) && IS_1_CMPLX(aBuffer->cmplxSame)) {
            tShard.RemoveAntiControl(&cShard);
        }
    }
}

} // namespace Qrack
