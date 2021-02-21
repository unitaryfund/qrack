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

#include <ctime>
#include <initializer_list>
#include <map>

#include "qfactory.hpp"
#include "qunit.hpp"

#define DIRTY(shard) (shard.isPhaseDirty || shard.isProbDirty)
#define IS_NORM_0(c) (c == ZERO_CMPLX)
#define IS_0_R1(r) (r == ZERO_R1)
#define IS_1_R1(r) (r == ONE_R1)
#define IS_1_CMPLX(c) (c == ONE_CMPLX)
#define SHARD_STATE(shard) (norm(shard.amp0) < (ONE_R1 / 2))
#define QUEUED_PHASE(shard)                                                                                            \
    ((shard.targetOfShards.size() != 0) || (shard.controlsShards.size() != 0) ||                                       \
        (shard.antiTargetOfShards.size() != 0) || (shard.antiControlsShards.size() != 0))
#define CACHED_Z(shard) (!shard.isPauliX && !shard.isPauliY && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_ZERO_OR_ONE(shard) (CACHED_Z(shard) && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1)))
#define CACHED_ZERO(shard) (CACHED_Z(shard) && IS_NORM_0(shard.amp1))
#define CACHED_ONE(shard) (CACHED_Z(shard) && IS_NORM_0(shard.amp0))
#define CACHED_X(shard) (shard.isPauliX && !DIRTY(shard) && !QUEUED_PHASE(shard))
#define CACHED_PLUS_OR_MINUS(shard) (CACHED_X(shard) && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1)))
#define CACHED_PLUS(shard) (CACHED_X(shard) && IS_NORM_0(shard.amp1))
/* "UNSAFE" variants here do not check whether the bit is in |0>/|1> rather than |+>/|-> basis. */
#define UNSAFE_CACHED_ZERO_OR_ONE(shard)                                                                               \
    (!shard.isProbDirty && !shard.isPauliX && !shard.isPauliY && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1)))
#define UNSAFE_CACHED_X(shard)                                                                                         \
    (!shard.isProbDirty && shard.isPauliX && !shard.isPauliY && (IS_NORM_0(shard.amp0) || IS_NORM_0(shard.amp1)))
#define UNSAFE_CACHED_ONE(shard) (!shard.isProbDirty && !shard.isPauliX && !shard.isPauliY && IS_NORM_0(shard.amp0))
#define UNSAFE_CACHED_ZERO(shard) (!shard.isProbDirty && !shard.isPauliX && !shard.isPauliY && IS_NORM_0(shard.amp1))
#define IS_SAME_UNIT(shard1, shard2) (shard1.unit && (shard1.unit == shard2.unit))
#define ARE_CLIFFORD(shard1, shard2)                                                                                   \
    ((engine == QINTERFACE_STABILIZER_HYBRID) && shard1.isClifford() && shard2.isClifford())

namespace Qrack {

QUnit::QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList,
    bitLenInt qubitThreshold)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , engine(eng)
    , subEngine(subEng)
    , devID(deviceID)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , freezeBasisH(false)
    , freezeBasis2Qb(false)
    , freezeClifford(false)
    , thresholdQubits(qubitThreshold)
{
    if ((engine == QINTERFACE_CPU) || (engine == QINTERFACE_OPENCL)) {
        subEngine = engine;
    }

    shards = QEngineShardMap();

    bool bitState;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((initState >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards.push_back(QEngineShard(bitState, doNormalize ? amplitudeFloor : ZERO_R1, GetNonunitaryPhase()));
    }
}

QInterfacePtr QUnit::MakeEngine(bitLenInt length, bitCapInt perm)
{
    return CreateQuantumInterface(engine, subEngine, length, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, amplitudeFloor, std::vector<int>{}, thresholdQubits);
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool bitState;

    Dump();

    shards = QEngineShardMap();

    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((perm >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards.push_back(QEngineShard(bitState, doNormalize ? amplitudeFloor : ZERO_R1, GetNonunitaryPhase()));
    }
}

void QUnit::SetQuantumState(const complex* inputState)
{
    Dump();

    if (qubitCount == 1U) {
        QEngineShard& shard = shards[0];
        shard.unit = NULL;
        shard.isProbDirty = false;
        shard.isPhaseDirty = false;
        shard.amp0 = inputState[0];
        shard.amp1 = inputState[1];
        shard.isPauliX = false;
        shard.isPauliY = false;
        if (IS_NORM_0(shard.amp0 - shard.amp1)) {
            shard.isPauliX = true;
            shard.isPauliY = false;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_NORM_0(shard.amp0 + shard.amp1)) {
            shard.isPauliX = true;
            shard.isPauliY = false;
            shard.amp1 = shard.amp0 / abs(shard.amp0);
            shard.amp0 = ZERO_R1;
        } else if (IS_NORM_0((I_CMPLX * inputState[0]) - inputState[1])) {
            shard.isPauliX = false;
            shard.isPauliY = true;
            shard.amp0 = shard.amp0 / abs(shard.amp0);
            shard.amp1 = ZERO_R1;
        } else if (IS_NORM_0((I_CMPLX * inputState[0]) + inputState[1])) {
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
        shards[idx] = QEngineShard(unit, idx, doNormalize ? amplitudeFloor : ZERO_R1);
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
        ToPermBasisAll();
        OrderContiguous(shards[0].unit);
        thisCopy = this;
    } else {
        thisCopyShared = std::dynamic_pointer_cast<QUnit>(Clone());
        thisCopyShared->EntangleAll();
        thisCopy = thisCopyShared.get();
    }

    thisCopy->shards[0].unit->GetProbs(outputProbs);
}

complex QUnit::GetAmplitude(bitCapInt perm)
{
    ToPermBasisAll();
    EndAllEmulation();

    complex result(ONE_R1, ZERO_R1);

    std::map<QInterfacePtr, bitCapInt> perms;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (perms.find(shards[i].unit) == perms.end()) {
            perms[shards[i].unit] = 0U;
        }
        if ((perm >> (bitCapIntOcl)i) & ONE_BCI) {
            perms[shards[i].unit] |= pow2(shards[i].mapped);
        }
    }

    for (auto&& qi : perms) {
        result *= qi.first->GetAmplitude(qi.second);
        if (IS_NORM_0(result)) {
            break;
        }
    }

    if ((shards[0].GetQubitCount() > 1) && IS_1_R1(norm(result)) && (randGlobalPhase || (result == ONE_CMPLX))) {
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
    /* TODO: This method should decompose the bits for the destination without composing the length first */

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
    std::map<QInterfacePtr, bitLenInt>::iterator subunit;
    for (subunit = subunits.begin(); subunit != subunits.end(); subunit++) {
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
        subunit = subunits.find(shard.unit);
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
        EndEmulation(shards[**bit]);
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
    for (bitLenInt i = 0; i < ebits.size(); i++) {
        ebits[i] = &bits[i];
    }

    return Entangle(ebits);
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt*> bits)
{
    for (bitLenInt i = 0; i < bits.size(); i++) {
        ToPermBasis(*(bits[i]));
    }
    return EntangleInCurrentBasis(bits.begin(), bits.end());
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start, bitLenInt length)
{
    ToPermBasis(start, length);

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

bool QUnit::TrySeparate(bitLenInt start, bitLenInt length, real1_f error_tol)
{
    if (length > 1) {
        QInterfacePtr dest = std::make_shared<QUnit>(
            engine, subEngine, length, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, useHostRam);

        if (TryDecompose(start, dest, error_tol)) {
            Compose(dest, start);
            return true;
        }

        return false;
    }

    // Otherwise, we're trying to separate a single bit.
    QEngineShard& shard = shards[start];

    if (shard.GetQubitCount() == 1) {
        return true;
    }

    if (shard.unit->isClifford()) {
        return TrySeparateCliffordBit(start);
    }

    // We check Z basis:
    real1 prob = ProbBase(start);
    bool didSeparate = (IS_0_R1(prob) || IS_1_R1(prob));

    // If this is 0.5, it wasn't Z basis, but it's worth checking X basis.
    if (!IS_0_R1(prob - ONE_R1 / 2)) {
        return didSeparate;
    }

    // We check X basis:
    shard.unit->H(shard.mapped);
    prob = ProbBase(start);
    didSeparate |= (IS_0_R1(prob) || IS_1_R1(prob));

    if (didSeparate || !IS_0_R1(prob - ONE_R1 / 2)) {
        H(start);
        return didSeparate;
    }

    // We check Y basis:
    complex mtrx[4] = { complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
        complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2) };
    shard.unit->ApplySingleBit(mtrx, shard.mapped);
    prob = ProbBase(start);
    didSeparate |= (IS_0_R1(prob) || IS_1_R1(prob));

    H(start);
    S(start);

    return didSeparate;
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
bool QUnit::CheckBitPermutation(const bitLenInt& qubitIndex, const bool& inCurrentBasis)
{
    if (!inCurrentBasis) {
        ToPermBasis(qubitIndex);
    }
    QEngineShard& shard = shards[qubitIndex];
    if (UNSAFE_CACHED_ZERO_OR_ONE(shard)) {
        return true;
    } else {
        return false;
    }
}

/// Check if all qubits in the range have cached probabilities indicating that they are in permutation basis
/// eigenstates, for optimization.
bool QUnit::CheckBitsPermutation(const bitLenInt& start, const bitLenInt& length, const bool& inCurrentBasis)
{
    // Certain optimizations become obvious, if all bits in a range are in permutation basis eigenstates.
    // Then, operations can often be treated as classical, instead of quantum.
    for (bitLenInt i = 0; i < length; i++) {
        if (!CheckBitPermutation(start + i, inCurrentBasis)) {
            return false;
        }
    }
    return true;
}

/// Assuming all bits in the range are in cached |0>/|1> eigenstates, read the unsigned integer value of the range.
bitCapInt QUnit::GetCachedPermutation(const bitLenInt& start, const bitLenInt& length)
{
    bitCapInt res = 0U;
    for (bitLenInt i = 0; i < length; i++) {
        if (SHARD_STATE(shards[start + i])) {
            res |= pow2(i);
        }
    }
    return res;
}

bitCapInt QUnit::GetCachedPermutation(const bitLenInt* bitArray, const bitLenInt& length)
{
    bitCapInt res = 0U;
    for (bitLenInt i = 0; i < length; i++) {
        if (SHARD_STATE(shards[bitArray[i]])) {
            res |= pow2(i);
        }
    }
    return res;
}

bool QUnit::CheckBitsPlus(const bitLenInt& qubitIndex, const bitLenInt& length)
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

real1_f QUnit::ProbBase(const bitLenInt& qubit)
{
    QEngineShard& shard = shards[qubit];

    if (!shard.isProbDirty) {
        return norm(shard.amp1);
    }

    shard.isProbDirty = false;

    bitLenInt shardQbCount = shard.GetQubitCount();
    QInterfacePtr unit = shard.unit;
    bitLenInt mapped = shard.mapped;
    real1 prob = unit->Prob(mapped);
    shard.amp1 = complex(sqrt(prob), ZERO_R1);
    shard.amp0 = complex(sqrt(ONE_R1 - prob), ZERO_R1);

    if (unit && unit->isClifford() && !TrySeparateCliffordBit(qubit)) {
        return prob;
    }

    bool didSeparate = false;
    if (IS_NORM_0(shard.amp1)) {
        SeparateBit(false, qubit);
        didSeparate = true;
    } else if (IS_NORM_0(shard.amp0)) {
        SeparateBit(true, qubit);
        didSeparate = true;
    }

    if (!didSeparate || (shardQbCount != 2)) {
        return prob;
    }

    bitLenInt partnerIndex;
    for (partnerIndex = 0; partnerIndex < qubitCount; partnerIndex++) {
        QEngineShard& partnerShard = shards[partnerIndex];
        if (unit == partnerShard.unit) {
            break;
        }
    }

    RevertBasis1Qb(partnerIndex);

    QEngineShard& partnerShard = shards[partnerIndex];

    complex amps[2];
    partnerShard.unit->GetQuantumState(amps);
    if (IS_NORM_0(amps[0] - amps[1])) {
        partnerShard.isPauliX = true;
        partnerShard.isPauliY = false;
        amps[0] = amps[0] / abs(amps[0]);
        amps[1] = ZERO_CMPLX;
    } else if (IS_NORM_0(amps[0] + amps[1])) {
        partnerShard.isPauliX = true;
        partnerShard.isPauliY = false;
        amps[1] = amps[0] / abs(amps[0]);
        amps[0] = ZERO_CMPLX;
    } else if (IS_NORM_0((I_CMPLX * amps[0]) - amps[1])) {
        shard.isPauliX = false;
        shard.isPauliY = true;
        amps[0] = amps[0] / abs(amps[0]);
        amps[1] = ZERO_CMPLX;
    } else if (IS_NORM_0((I_CMPLX * amps[0]) + amps[1])) {
        shard.isPauliX = false;
        shard.isPauliY = true;
        amps[1] = amps[0] / abs(amps[0]);
        amps[0] = ZERO_CMPLX;
    }
    partnerShard.amp0 = amps[0];
    partnerShard.amp1 = amps[1];
    partnerShard.isProbDirty = false;
    partnerShard.isPhaseDirty = false;
    partnerShard.unit = NULL;
    if (doNormalize) {
        partnerShard.ClampAmps(amplitudeFloor);
    }

    return prob;
}

bool QUnit::TrySeparateCliffordBit(const bitLenInt& qubit)
{
    QEngineShard& shard = shards[qubit];

    if (shard.GetQubitCount() == 1) {
        return true;
    }

    if (freezeClifford || !shard.unit->isClifford()) {
        return false;
    }

    freezeClifford = true;

    QInterfacePtr unit = shards[qubit].unit;

    ProbBase(qubit);

    if (IS_NORM_0(shard.amp1)) {
        SeparateBit(false, qubit);
    } else if (IS_NORM_0(shard.amp0)) {
        SeparateBit(true, qubit);
    } else if (!unit->TrySeparate(shard.mapped)) {
        return false;
    } else {
        unit->H(shard.mapped);
        ProbBase(qubit);

        if (IS_NORM_0(shard.amp1)) {
            SeparateBit(false, qubit);
            H(qubit);
        } else if (IS_NORM_0(shard.amp0)) {
            SeparateBit(true, qubit);
            H(qubit);
        } else {
            unit->H(shard.mapped);

            unit->S(shard.mapped);
            unit->H(shard.mapped);
            ProbBase(qubit);

            if (IS_NORM_0(shard.amp1)) {
                SeparateBit(false, qubit);
                H(qubit);
                IS(qubit);
            } else if (IS_NORM_0(shard.amp0)) {
                SeparateBit(true, qubit);
                H(qubit);
                IS(qubit);
            } else {
                unit->H(shard.mapped);
                unit->IS(shard.mapped);
                ProbBase(qubit);
                return false;
            }
        }
    }

    return true;
}

real1_f QUnit::Prob(bitLenInt qubit)
{
    ToPermBasis(qubit);
    return ProbBase(qubit);
}

real1_f QUnit::ProbAll(bitCapInt perm) { return clampProb(norm(GetAmplitude(perm))); }

real1_f QUnit::ProbParity(const bitCapInt& mask)
{
    // If no bits in mask:
    if (!mask) {
        return ZERO_R1;
    }

    // If only one bit in mask:
    if (!(mask & (mask - ONE_BCI))) {
        return Prob(log2(mask));
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
    }

    std::map<QInterfacePtr, bitCapInt> units;
    real1 oddChance = ZERO_R1;
    real1 nOddChance;
    for (bitLenInt i = 0; i < qIndices.size(); i++) {
        ToPermBasis(qIndices[i]);
        QEngineShard& shard = shards[qIndices[i]];
        if (!(shard.unit)) {
            nOddChance = shard.Prob();
            oddChance = (oddChance * (ONE_R1 - nOddChance)) + ((ONE_R1 - oddChance) * nOddChance);
        } else if (units.find(shard.unit) == units.end()) {
            units[shard.unit] = pow2(shard.mapped);
        } else {
            units[shard.unit] |= pow2(shard.mapped);
        }
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

bool QUnit::ForceMParity(const bitCapInt& mask, bool result, bool doForce)
{
    // If no bits in mask:
    if (!mask) {
        return false;
    }

    // If only one bit in mask:
    if (!(mask & (mask - ONE_BCI))) {
        return ForceM(log2(mask), result, doForce);
    }

    bitCapInt nV = mask;
    std::vector<bitLenInt> qIndices;
    for (bitCapInt v = mask; v; v = nV) {
        nV &= (v - ONE_BCI); // clear the least significant bit set
        qIndices.push_back(log2((v ^ nV) & v));
    }

    bool flipResult = false;
    std::vector<bitLenInt> eIndices;
    for (bitLenInt i = 0; i < qIndices.size(); i++) {
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
    for (bitLenInt i = 0; i < eIndices.size(); i++) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    return flipResult ^ (unit->ForceMParity(mappedMask, result ^ flipResult, doForce));
}

void QUnit::SeparateBit(bool value, bitLenInt qubit, bool doDispose)
{
    QInterfacePtr unit = shards[qubit].unit;

    if (unit == NULL) {
        return;
    }

    bitLenInt mapped = shards[qubit].mapped;

    shards[qubit].unit = NULL;
    shards[qubit].mapped = 0;
    shards[qubit].isProbDirty = false;
    shards[qubit].isPhaseDirty = false;
    shards[qubit].amp0 = value ? ZERO_CMPLX : GetNonunitaryPhase();
    shards[qubit].amp1 = value ? GetNonunitaryPhase() : ZERO_CMPLX;

    if (!doDispose || !unit || (unit->GetQubitCount() == 1)) {
        return;
    }

    unit->Dispose(mapped, 1, value ? ONE_BCI : 0);

    /* Update the mappings. */
    for (auto&& shard : shards) {
        if ((shard.unit == unit) && (shard.mapped > mapped)) {
            shard.mapped--;
        }
    }
}

bool QUnit::ForceM(bitLenInt qubit, bool res, bool doForce, bool doApply)
{
    ToPermBasis(qubit);

    QEngineShard& shard = shards[qubit];

    bool result;
    if (!shard.isProbDirty && !shard.unit) {
        result = doForce ? res : (Rand() <= norm(shard.amp1));
    } else if (shard.unit->isClifford()) {
        real1 prob = shard.Prob();
        if (prob == ZERO_R1) {
            result = false;
        } else if (prob == ONE_R1) {
            result = true;
        } else {
            result = shard.unit->ForceM(shard.mapped, res, doForce, doApply);
        }
    } else {
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
        return result;
    }

    // This is critical: it's the "nonlocal correlation" of "wave function collapse".
    if (shard.unit) {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if ((i != qubit) && shards[i].unit == shard.unit) {
                shards[i].MakeDirty();
            }
        }
        if (!shard.unit->isClifford() || shard.unit->TrySeparate(qubit)) {
            SeparateBit(result, qubit);
        }
    }

    return result;
}

bitCapInt QUnit::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    if (!doForce && doApply && (length == qubitCount) && (engine == QINTERFACE_STABILIZER_HYBRID)) {
        return MAll();
    }

    // This will discard all buffered gates that don't affect Z basis probability,
    // so it's safe to call ToPermBasis() without performance penalty, afterward.
    ToPermBasisMeasure(start, length);

    return QInterface::ForceMReg(start, length, result, doForce, doApply);
}

bitCapInt QUnit::MAll()
{
    if (engine != QINTERFACE_STABILIZER_HYBRID) {
        return MReg(0, qubitCount);
    }

    ToPermBasisAllMeasure();

    std::vector<bitCapInt> partResults;
    bitCapInt toRet = 0;

    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (!toFind) {
            if (Rand() <= norm(shards[i].amp1)) {
                shards[i].amp0 = ZERO_CMPLX;
                shards[i].amp1 = GetNonunitaryPhase();
                toRet |= pow2(i);
            } else {
                shards[i].amp0 = GetNonunitaryPhase();
                shards[i].amp1 = ZERO_CMPLX;
            }
        } else if (!(toFind->isClifford())) {
            if (M(i)) {
                toRet |= pow2(i);
            }
        } else if (find(units.begin(), units.end(), toFind) == units.end()) {
            units.push_back(toFind);
            partResults.push_back(toFind->MAll());
        }
    }

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (!shards[i].unit) {
            continue;
        }
        bitLenInt offset = find(units.begin(), units.end(), shards[i].unit) - units.begin();
        if (offset < partResults.size()) {
            toRet |= ((partResults[offset] >> shards[i].mapped) & 1U) << i;
        }
    }

    SetPermutation(toRet);

    return toRet;
}

/// Set register bits to given permutation
void QUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    MReg(start, length);

    bool bitState;
    for (bitLenInt i = 0; i < length; i++) {
        bitState = ((value >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards[i + start] = QEngineShard(bitState, doNormalize ? amplitudeFloor : ZERO_R1, GetNonunitaryPhase());
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
    ApplyAntiControlledSinglePhase(control, 1U, qubit2, ONE_CMPLX, I_CMPLX);
    control[0] = qubit2;
    ApplyAntiControlledSinglePhase(control, 1U, qubit1, ONE_CMPLX, I_CMPLX);

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
    real1 sinTheta = sin(theta);

    if (IS_0_R1(sinTheta)) {
        ApplyControlledSinglePhase(controls, 1, qubit2, ONE_CMPLX, exp(complex(ZERO_R1, phi)));
        return;
    }

    if (IS_1_R1(-sinTheta)) {
        ISwap(qubit1, qubit2);
        ApplyControlledSinglePhase(controls, 1, qubit2, ONE_CMPLX, exp(complex(ZERO_R1, phi)));
        return;
    }

    RevertBasis1Qb(qubit1);
    RevertBasis1Qb(qubit2);
    RevertBasis2Qb(qubit1, ONLY_INVERT);
    RevertBasis2Qb(qubit2, ONLY_INVERT);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (UNSAFE_CACHED_ZERO_OR_ONE(shard1) && UNSAFE_CACHED_ZERO_OR_ONE(shard2) &&
        (SHARD_STATE(shard1) == SHARD_STATE(shard2))) {
        // We can avoid dirtying the cache and entangling, since this gate doesn't swap identical classical bits.
        if (SHARD_STATE(shard1)) {
            ApplyControlledSinglePhase(controls, 1, qubit2, ONE_CMPLX, exp(complex(ZERO_R1, phi)));
        }
        return;
    }

    Entangle({ qubit1, qubit2 })->FSim(theta, phi, shard1.mapped, shard2.mapped);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.MakeDirty();
    shard2.MakeDirty();
}

void QUnit::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        ApplySinglePhase(-ONE_CMPLX, ONE_CMPLX, start);
        return;
    }

    if ((engine == QINTERFACE_QPAGER) || (subEngine == QINTERFACE_QPAGER)) {
        // TODO: Case below this should work for QPager, but doesn't
        EntangleRange(start, length);
        shards[start].unit->ZeroPhaseFlip(shards[start].mapped, length);
        DirtyShardRange(start, length);
        return;
    }

    QInterface::ZeroPhaseFlip(start, length);
}

void QUnit::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    // If there are no controls, this is equivalent to the single bit gate.
    if (!controlLen) {
        ApplySingleBit(mtrxs, qubitIndex);
        return;
    }

    bitLenInt i;

    std::vector<bitLenInt> trimmedControls;
    std::vector<bitCapInt> skipPowers;
    bitCapInt skipValueMask = 0;
    for (i = 0; i < controlLen; i++) {
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
        ApplySingleBit(mtrx, qubitIndex);
        return;
    }

    std::vector<bitLenInt> bits(trimmedControls.size() + 1);
    for (i = 0; i < trimmedControls.size(); i++) {
        bits[i] = trimmedControls[i];
    }
    bits[trimmedControls.size()] = qubitIndex;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(trimmedControls.size() + 1);
    for (i = 0; i < bits.size(); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    bitLenInt* mappedControls = new bitLenInt[trimmedControls.size()];
    for (i = 0; i < trimmedControls.size(); i++) {
        mappedControls[i] = shards[trimmedControls[i]].mapped;
        shards[trimmedControls[i]].isPhaseDirty = true;
    }

    unit->UniformlyControlledSingleBit(mappedControls, trimmedControls.size(), shards[qubitIndex].mapped, mtrxs,
        &(skipPowers[0]), skipPowers.size(), skipValueMask);

    shards[qubitIndex].MakeDirty();

    delete[] mappedControls;
}

void QUnit::CUniformParityRZ(
    const bitLenInt* cControls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
{
    std::vector<bitLenInt> controls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        QEngineShard& shard = shards[cControls[i]];

        if (!CACHED_Z(shard)) {
            // Control becomes entangled
            controls.push_back(cControls[i]);
            continue;
        }

        if (IS_NORM_0(shard.amp1)) {
            // Gate does nothing
            return;
        }

        if (!IS_NORM_0(shard.amp0)) {
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
    for (bitLenInt i = 0; i < qIndices.size(); i++) {
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
        real1 cosine = cos(angle);
        real1 sine = sin(angle);
        complex phaseFac;
        if (flipResult) {
            phaseFac = complex(cosine, sine);
        } else {
            phaseFac = complex(cosine, -sine);
        }
        if (controls.size() == 0) {
            return ApplySinglePhase(phaseFac, phaseFac, 0);
        } else {
            return ApplyControlledSinglePhase(&(controls[0]), controls.size(), 0, phaseFac, phaseFac);
        }
    }

    if (eIndices.size() == 1U) {
        real1 cosine = cos(angle);
        real1 sine = sin(angle);
        complex phaseFac, phaseFacAdj;
        if (flipResult) {
            phaseFac = complex(cosine, -sine);
            phaseFacAdj = complex(cosine, sine);
        } else {
            phaseFac = complex(cosine, sine);
            phaseFacAdj = complex(cosine, -sine);
        }
        if (controls.size() == 0) {
            return ApplySinglePhase(phaseFacAdj, phaseFac, eIndices[0]);
        } else {
            return ApplyControlledSinglePhase(&(controls[0]), controls.size(), eIndices[0], phaseFacAdj, phaseFac);
        }
    }

    for (bitLenInt i = 0; i < eIndices.size(); i++) {
        shards[eIndices[i]].isPhaseDirty = true;
    }

    QInterfacePtr unit = Entangle(eIndices);

    bitCapInt mappedMask = 0;
    for (bitLenInt i = 0; i < eIndices.size(); i++) {
        mappedMask |= pow2(shards[eIndices[i]].mapped);
    }

    if (controls.size() == 0) {
        unit->UniformParityRZ(mappedMask, flipResult ? -angle : angle);
    } else {
        std::vector<bitLenInt*> ebits(controls.size());
        for (bitLenInt i = 0; i < controls.size(); i++) {
            ebits[i] = &controls[i];
        }

        Entangle(ebits);
        unit = Entangle({ controls[0], eIndices[0] });

        std::vector<bitLenInt> controlsMapped(controls.size());
        for (bitLenInt i = 0; i < controls.size(); i++) {
            QEngineShard& cShard = shards[controls[i]];
            controlsMapped[i] = cShard.mapped;
            cShard.isPhaseDirty = true;
        }

        unit->CUniformParityRZ(&(controlsMapped[0]), controlsMapped.size(), mappedMask, flipResult ? -angle : angle);
    }
}

void QUnit::H(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (!freezeBasisH) {
        RevertBasisY(target);
        CommuteH(target);
        shard.isPauliX = !shard.isPauliX;
        return;
    }

    if (shard.unit) {
        shard.unit->H(shard.mapped);
    }
    if (DIRTY(shard)) {
        shard.MakeDirty();
        return;
    }

    complex tempAmp1 = ((real1)M_SQRT1_2) * (shard.amp0 - shard.amp1);
    shard.amp0 = ((real1)M_SQRT1_2) * (shard.amp0 + shard.amp1);
    shard.amp1 = tempAmp1;
    if (doNormalize) {
        shard.ClampAmps(amplitudeFloor);
    }
}

void QUnit::XBase(const bitLenInt& target)
{
    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->X(shard.mapped);
    }
    if (DIRTY(shard)) {
        shard.MakeDirty();
        return;
    }

    std::swap(shard.amp0, shard.amp1);
}

void QUnit::YBase(const bitLenInt& target)
{
    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->Y(shard.mapped);
    }
    if (DIRTY(shard)) {
        shard.MakeDirty();
        return;
    }

    complex Y0 = shard.amp0;
    shard.amp0 = -I_CMPLX * shard.amp1;
    shard.amp1 = I_CMPLX * Y0;
}

void QUnit::ZBase(const bitLenInt& target)
{
    QEngineShard& shard = shards[target];

    if (shard.unit) {
        shard.unit->Z(shard.mapped);
    }
    if (DIRTY(shard)) {
        shard.MakeDirty();
        return;
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

    if (shard.IsInvertTarget()) {
        RevertBasis1Qb(target);
        shard.CommutePhase(ONE_CMPLX, -ONE_CMPLX);
    } else {
        if (UNSAFE_CACHED_ZERO(shard)) {
            Flush0Eigenstate(target);
            return;
        }
    }

    if (shard.isPauliX || shard.isPauliY) {
        XBase(target);
    } else if (!shard.isPauliY) {
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

void QUnit::TransformXInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut)
{
    mtrxOut[0] = (real1)(ONE_R1 / 2) * (complex)(topRight + bottomLeft);
    mtrxOut[1] = (real1)(ONE_R1 / 2) * (complex)(-topRight + bottomLeft);
    mtrxOut[2] = -mtrxOut[1];
    mtrxOut[3] = -mtrxOut[0];
}

void QUnit::TransformY2x2(const complex* mtrxIn, complex* mtrxOut)
{
    mtrxOut[0] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + I_CMPLX * (mtrxIn[1] - mtrxIn[2]) + mtrxIn[3]);
    mtrxOut[1] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + I_CMPLX * (-mtrxIn[1] - mtrxIn[2]) - mtrxIn[3]);
    mtrxOut[2] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + I_CMPLX * (mtrxIn[1] + mtrxIn[2]) - mtrxIn[3]);
    mtrxOut[3] = (real1)(ONE_R1 / 2) * (complex)(mtrxIn[0] + I_CMPLX * (-mtrxIn[1] + mtrxIn[2]) + mtrxIn[3]);
}

void QUnit::TransformYInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut)
{
    mtrxOut[0] = I_CMPLX * (real1)(ONE_R1 / 2) * (complex)(topRight - bottomLeft);
    mtrxOut[1] = I_CMPLX * (real1)(ONE_R1 / 2) * (complex)(-topRight - bottomLeft);
    mtrxOut[2] = -mtrxOut[1];
    mtrxOut[3] = -mtrxOut[0];
}

void QUnit::TransformPhase(const complex& topLeft, const complex& bottomRight, complex* mtrxOut)
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
        [&]() { bare; });

#define CTRLED_SWAP_WRAP(ctrld, bare, anti)                                                                            \
    if (qubit1 == qubit2) {                                                                                            \
        return;                                                                                                        \
    }                                                                                                                  \
    ToPermBasis(qubit1);                                                                                               \
    ToPermBasis(qubit2);                                                                                               \
    ApplyEitherControlled(                                                                                             \
        controls, controlLen, { qubit1, qubit2 }, anti,                                                                \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->ctrld; }, [&]() { bare; })
#define CTRL_GEN_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, trnsMtrx
#define CTRL_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, mtrx
#define CTRL_1_ARGS mappedControls[0], shards[target].mapped
#define CTRL_2_ARGS mappedControls[0], mappedControls[1], shards[target].mapped
#define CTRL_S_ARGS &(mappedControls[0]), mappedControls.size(), shards[qubit1].mapped, shards[qubit2].mapped
#define CTRL_P_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, topLeft, bottomRight
#define CTRL_I_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, topRight, bottomLeft

void QUnit::CNOT(bitLenInt control, bitLenInt target)
{
    QEngineShard& tShard = shards[target];

    if (CACHED_X(tShard)) {
        if (IS_NORM_0(tShard.amp1)) {
            return;
        }
        if (IS_NORM_0(tShard.amp0)) {
            Z(control);
            return;
        }
    }

    QEngineShard& cShard = shards[control];

    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (IS_NORM_0(cShard.amp1)) {
            Flush0Eigenstate(control);
            return;
        }
        if (IS_NORM_0(cShard.amp0)) {
            Flush1Eigenstate(control);
            X(target);
            return;
        }
    }

    RevertBasisY(target);

    bool pmBasis = (cShard.isPauliX && tShard.isPauliX && !QUEUED_PHASE(cShard) && !QUEUED_PHASE(tShard));

    if (!freezeBasis2Qb && !pmBasis) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && !ARE_CLIFFORD(cShard, tShard)) {
            tShard.AddInversionAngles(&cShard, ONE_CMPLX, ONE_CMPLX);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;

    // We're free to transform gates to any orthonormal basis of the Hilbert space.
    // For a 2 qubit system, if the control is the lefthand bit, it's easy to verify the following truth table for CNOT:
    // |++> -> |++>
    // |+-> -> |-->
    // |-+> -> |-+>
    // |--> -> |+->
    // Under the Jacobian transformation between these two bases for defining the truth table, the matrix representation
    // is equivalent to the gate with bits flipped. We just let ApplyEitherControlled() know to leave the current basis
    // alone, by way of the last optional "true" argument in the call.
    if (pmBasis) {
        std::swap(controls[0], target);
        ApplyEitherControlled(
            controls, controlLen, { target }, false,
            [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->CNOT(CTRL_1_ARGS); },
            [&]() { XBase(target); }, true);
        return;
    }

    CTRLED_PHASE_INVERT_WRAP(
        CNOT(CTRL_1_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS), X(target), false, true, ONE_CMPLX, ONE_CMPLX);
}

void QUnit::AntiCNOT(bitLenInt control, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    if (CACHED_PLUS(tShard)) {
        return;
    }

    QEngineShard& cShard = shards[control];
    if (!cShard.IsInvertTarget() && UNSAFE_CACHED_ZERO_OR_ONE(cShard)) {
        if (IS_NORM_0(cShard.amp1)) {
            Flush0Eigenstate(control);
            X(target);
            return;
        }
        if (IS_NORM_0(cShard.amp0)) {
            Flush1Eigenstate(control);
            return;
        }
    }

    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;

    if (!freezeBasis2Qb) {
        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, INVERT_AND_PHASE, CONTROLS_AND_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && !ARE_CLIFFORD(cShard, tShard)) {
            shards[target].AddAntiInversionAngles(&(shards[control]), ONE_CMPLX, ONE_CMPLX);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(AntiCNOT(CTRL_1_ARGS), ApplyAntiControlledSingleBit(CTRL_GEN_ARGS), X(target), true, true,
        ONE_CMPLX, ONE_CMPLX);
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
            if (IS_NORM_0(c1Shard.amp1)) {
                Flush0Eigenstate(control1);
                return;
            }
            if (IS_NORM_0(c1Shard.amp0)) {
                Flush1Eigenstate(control1);
                CNOT(control2, target);
                return;
            }
        }
    }

    if (!c2Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c2Shard)) {
            if (IS_NORM_0(c2Shard.amp1)) {
                Flush0Eigenstate(control2);
                return;
            }
            if (IS_NORM_0(c2Shard.amp0)) {
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

    bitLenInt controls[2] = { control1, control2 };

    ApplyEitherControlled(
        controls, 2, { target }, false,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliY) {
                unit->ApplyControlledSingleInvert(
                    &(mappedControls[0]), mappedControls.size(), shards[target].mapped, -I_CMPLX, I_CMPLX);
            } else if (shards[target].isPauliX) {
                if (mappedControls.size() == 2) {
                    unit->CCZ(CTRL_2_ARGS);
                } else {
                    unit->CZ(CTRL_1_ARGS);
                }
            } else {
                if (mappedControls.size() == 2) {
                    unit->CCNOT(CTRL_2_ARGS);
                } else {
                    unit->CNOT(CTRL_1_ARGS);
                }
            }
        },
        [&]() { X(target); });
}

void QUnit::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    QEngineShard& tShard = shards[target];
    if (CACHED_PLUS(tShard)) {
        return;
    }

    bitLenInt controls[2] = { control1, control2 };

    ApplyEitherControlled(
        controls, 2, { target }, true,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPauliY) {
                unit->ApplyAntiControlledSingleInvert(
                    &(mappedControls[0]), mappedControls.size(), shards[target].mapped, -I_CMPLX, I_CMPLX);
            } else if (shards[target].isPauliX) {
                unit->ApplyAntiControlledSinglePhase(
                    &(mappedControls[0]), mappedControls.size(), shards[target].mapped, ONE_CMPLX, -ONE_CMPLX);
            } else {
                if (mappedControls.size() == 2) {
                    unit->AntiCCNOT(CTRL_2_ARGS);
                } else {
                    unit->AntiCNOT(CTRL_1_ARGS);
                }
            }
        },
        [&]() { X(target); });
}

void QUnit::CZ(bitLenInt control, bitLenInt target)
{
    if (shards[control].isPauliX && !shards[target].isPauliX && !shards[target].isPauliY) {
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
        RevertBasis2Qb(target, ONLY_INVERT, ONLY_TARGETS, CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard) && !ARE_CLIFFORD(cShard, tShard)) {
            tShard.AddPhaseAngles(&cShard, ONE_CMPLX, -ONE_CMPLX);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(
        CZ(CTRL_1_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS), Z(target), false, false, ONE_CMPLX, -ONE_CMPLX);
}

void QUnit::CH(bitLenInt control, bitLenInt target)
{
    const complex mtrx[4] = { complex(ONE_R1 / sqrt((real1)2), ZERO_R1), complex(ONE_R1 / sqrt((real1)2), ZERO_R1),
        complex(ONE_R1 / sqrt((real1)2), ZERO_R1), complex(-ONE_R1 / sqrt((real1)2), ZERO_R1) };

    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;

    CTRLED_GEN_WRAP(ApplyControlledSingleBit(CTRL_GEN_ARGS), H(target), false);
}

void QUnit::CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    if (shards[control1].isPauliX && !shards[target].isPauliX && !shards[target].isPauliY) {
        std::swap(control1, target);
    }

    if (shards[control2].isPauliX && !shards[target].isPauliX && !shards[target].isPauliY) {
        std::swap(control2, target);
    }

    QEngineShard& tShard = shards[target];
    QEngineShard& c1Shard = shards[control1];
    QEngineShard& c2Shard = shards[control2];

    if (!c1Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c1Shard)) {
            if (IS_NORM_0(c1Shard.amp1)) {
                Flush0Eigenstate(control1);
                return;
            }
            if (IS_NORM_0(c1Shard.amp0)) {
                Flush1Eigenstate(control1);
                CZ(control2, target);
                return;
            }
        }
    }

    if (!c2Shard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(c2Shard)) {
            if (IS_NORM_0(c2Shard.amp1)) {
                Flush0Eigenstate(control2);
                return;
            }
            if (IS_NORM_0(c2Shard.amp0)) {
                Flush1Eigenstate(control2);
                CZ(control1, target);
                return;
            }
        }
    }

    if (!tShard.IsInvertTarget()) {
        if (UNSAFE_CACHED_ZERO_OR_ONE(tShard)) {
            if (IS_NORM_0(tShard.amp1)) {
                Flush0Eigenstate(target);
                return;
            }
            if (IS_NORM_0(tShard.amp0)) {
                Flush1Eigenstate(target);
                CZ(control1, control2);
                return;
            }
        }
    }

    bitLenInt controls[2] = { control1, control2 };

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
        [&]() { Z(target); });
}

void QUnit::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
{
    if (IS_NORM_0(topLeft - bottomRight) && (randGlobalPhase || IS_1_R1(topLeft))) {
        return;
    }

    if (IS_NORM_0(topLeft + bottomRight) && (randGlobalPhase || IS_1_R1(topLeft))) {
        Z(target);
        return;
    }

    QEngineShard& shard = shards[target];

    if (shard.IsInvertTarget()) {
        RevertBasis1Qb(target);
        shard.CommutePhase(topLeft, bottomRight);
    } else {
        if (IS_1_R1(topLeft) && UNSAFE_CACHED_ZERO(shard)) {
            Flush0Eigenstate(target);
            return;
        }

        if (IS_1_R1(bottomRight) && UNSAFE_CACHED_ONE(shard)) {
            Flush1Eigenstate(target);
            return;
        }
    }

    if (!freezeBasisH && shard.isPauliY) {
        if (randGlobalPhase || IS_1_R1(topLeft)) {
            if (IS_NORM_0((I_CMPLX * topLeft) - bottomRight)) {
                shard.isPauliX = true;
                shard.isPauliY = false;
                XBase(target);
                return;
            } else if (IS_NORM_0((I_CMPLX * topLeft) + bottomRight)) {
                shard.isPauliX = true;
                shard.isPauliY = false;
                return;
            }
        }

        complex mtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
        TransformPhase(topLeft, bottomRight, mtrx);

        if (shard.unit) {
            shard.unit->ApplySingleBit(mtrx, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.MakeDirty();
            return;
        }

        complex Y0 = shard.amp0;

        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }
    } else if (shard.isPauliX) {
        if (!freezeBasisH && (randGlobalPhase || IS_1_R1(topLeft))) {
            if (IS_NORM_0((I_CMPLX * topLeft) - bottomRight)) {
                shard.isPauliX = false;
                shard.isPauliY = true;
                return;
            } else if (IS_NORM_0((I_CMPLX * topLeft) + bottomRight)) {
                shard.isPauliX = false;
                shard.isPauliY = true;
                XBase(target);
                return;
            }
        }

        complex mtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
        TransformPhase(topLeft, bottomRight, mtrx);

        if (shard.unit) {
            shard.unit->ApplySingleBit(mtrx, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.MakeDirty();
            return;
        }

        complex Y0 = shard.amp0;

        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }
    } else {
        if (shard.unit) {
            shard.unit->ApplySinglePhase(topLeft, bottomRight, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.MakeDirty();
            return;
        }

        shard.amp0 *= topLeft;
        shard.amp1 *= bottomRight;
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }
    }
}

void QUnit::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
{
    if (IS_NORM_0(topRight - bottomLeft) && (randGlobalPhase || IS_1_CMPLX(topRight))) {
        X(target);
        return;
    }

    QEngineShard& shard = shards[target];

    if (shard.IsInvertTarget()) {
        RevertBasis1Qb(target);
        shard.CommutePhase(bottomLeft, topRight);
    }

    shard.FlipPhaseAnti();

    if (shard.isPauliX || shard.isPauliY) {
        complex mtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
        if (shard.isPauliX) {
            TransformXInvert(topRight, bottomLeft, mtrx);
        } else {
            TransformYInvert(topRight, bottomLeft, mtrx);
        }

        if (shard.unit) {
            shard.unit->ApplySingleBit(mtrx, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.MakeDirty();
            return;
        }

        complex Y0 = shard.amp0;

        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }
    } else {
        if (shard.unit) {
            shard.unit->ApplySingleInvert(topRight, bottomLeft, shard.mapped);
        }
        if (DIRTY(shard)) {
            shard.MakeDirty();
            return;
        }

        complex tempAmp1 = shard.amp0 * bottomLeft;
        shard.amp0 = shard.amp1 * topRight;
        shard.amp1 = tempAmp1;
        if (doNormalize) {
            shard.ClampAmps(amplitudeFloor);
        }
    }
}

void QUnit::ApplyControlledSinglePhase(const bitLenInt* cControls, const bitLenInt& controlLen,
    const bitLenInt& cTarget, const complex topLeft, const complex bottomRight)
{
    // Commutes with controlled phase optimizations
    if (!controlLen) {
        ApplySinglePhase(topLeft, bottomRight, cTarget);
        return;
    }

    if ((controlLen == 1) && IS_NORM_0(topLeft - bottomRight)) {
        ApplySinglePhase(ONE_CMPLX, bottomRight, cControls[0]);
        return;
    }

    bitLenInt* controls = new bitLenInt[controlLen];
    std::copy(cControls, cControls + controlLen, controls);
    bitLenInt target = cTarget;

    QEngineShard& shard = shards[target];

    if (IS_1_R1(bottomRight) && (!shard.IsInvertTarget() && UNSAFE_CACHED_ONE(shard))) {
        Flush1Eigenstate(target);
        delete[] controls;
        return;
    }

    if (IS_1_R1(topLeft)) {
        if (!shard.IsInvertTarget() && UNSAFE_CACHED_ZERO(shard)) {
            Flush0Eigenstate(target);
            delete[] controls;
            return;
        }

        if (IS_1_R1(-bottomRight)) {
            if (controlLen == 2U) {
                CCZ(controls[0], controls[1], target);
                delete[] controls;
                return;
            }
            if (controlLen == 1U) {
                CZ(controls[0], target);
                delete[] controls;
                return;
            }
        }

        if (!shards[target].isPauliX && !shards[target].isPauliY) {
            for (bitLenInt i = 0; i < controlLen; i++) {
                if (shards[controls[i]].isPauliX) {
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
                ApplySinglePhase(topLeft, bottomRight, target);
            } else {
                Flush0Eigenstate(control);
            }

            delete[] controls;
            return;
        }

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, IS_1_CMPLX(topLeft) ? ONLY_TARGETS : CONTROLS_AND_TARGETS, CTRL_AND_ANTI,
            {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard)) {
            delete[] controls;
            tShard.AddPhaseAngles(&cShard, topLeft, bottomRight);
            OptimizePairBuffers(control, target, false);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(ApplyControlledSinglePhase(CTRL_P_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS),
        ApplySinglePhase(topLeft, bottomRight, target), false, false, topLeft, bottomRight);

    delete[] controls;
}

void QUnit::ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex topRight, const complex bottomLeft)
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

    CTRLED_PHASE_INVERT_WRAP(ApplyControlledSingleInvert(CTRL_I_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS),
        ApplySingleInvert(topRight, bottomLeft, target), false, true, topRight, bottomLeft);
}

void QUnit::ApplyAntiControlledSinglePhase(const bitLenInt* cControls, const bitLenInt& controlLen,
    const bitLenInt& cTarget, const complex topLeft, const complex bottomRight)
{
    // Commutes with controlled phase optimizations
    if (!controlLen) {
        ApplySinglePhase(topLeft, bottomRight, cTarget);
        return;
    }

    if ((controlLen == 1) && IS_NORM_0(topLeft - bottomRight)) {
        ApplySinglePhase(topLeft, ONE_CMPLX, cControls[0]);
        return;
    }

    bitLenInt* controls = new bitLenInt[controlLen];
    std::copy(cControls, cControls + controlLen, controls);
    bitLenInt target = cTarget;

    QEngineShard& shard = shards[target];

    if (IS_1_R1(topLeft) && (!shard.IsInvertTarget() && UNSAFE_CACHED_ZERO(shard))) {
        Flush0Eigenstate(target);
        delete[] controls;
        return;
    }

    if (IS_1_R1(bottomRight)) {
        if (!shard.IsInvertTarget() && UNSAFE_CACHED_ONE(shard)) {
            Flush1Eigenstate(target);
            delete[] controls;
            return;
        }

        if (!shards[target].isPauliX && !shards[target].isPauliY) {
            for (bitLenInt i = 0; i < controlLen; i++) {
                if (shards[controls[i]].isPauliX) {
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
                ApplySinglePhase(topLeft, bottomRight, target);
            }
            delete[] controls;
            return;
        }

        RevertBasis2Qb(control, ONLY_INVERT, ONLY_TARGETS);
        RevertBasis2Qb(target, ONLY_INVERT, IS_1_CMPLX(bottomRight) ? ONLY_TARGETS : CONTROLS_AND_TARGETS,
            CTRL_AND_ANTI, {}, { control });

        if (!IS_SAME_UNIT(cShard, tShard)) {
            delete[] controls;
            tShard.AddAntiPhaseAngles(&cShard, bottomRight, topLeft);
            OptimizePairBuffers(control, target, true);

            return;
        }
    }

    CTRLED_PHASE_INVERT_WRAP(ApplyAntiControlledSinglePhase(CTRL_P_ARGS), ApplyAntiControlledSingleBit(CTRL_GEN_ARGS),
        ApplySinglePhase(topLeft, bottomRight, target), true, false, topLeft, bottomRight);

    delete[] controls;
}

void QUnit::ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    if ((controlLen == 1U) && IS_1_R1(topRight) && IS_1_R1(bottomLeft)) {
        AntiCNOT(controls[0], target);
        return;
    }

    CTRLED_PHASE_INVERT_WRAP(ApplyAntiControlledSingleInvert(CTRL_I_ARGS), ApplyAntiControlledSingleBit(CTRL_GEN_ARGS),
        ApplySingleInvert(topRight, bottomLeft, target), true, true, topRight, bottomLeft);
}

void QUnit::ApplySingleBit(const complex* mtrx, bitLenInt target)
{
    if (IsIdentity(mtrx, true)) {
        return;
    }

    QEngineShard& shard = shards[target];

    if (!norm(mtrx[1]) && !norm(mtrx[2])) {
        ApplySinglePhase(mtrx[0], mtrx[3], target);
        return;
    }
    if (!norm(mtrx[0]) && !norm(mtrx[3])) {
        ApplySingleInvert(mtrx[1], mtrx[2], target);
        return;
    }
    if (!shard.isPauliY && (randGlobalPhase || (mtrx[0] == complex(M_SQRT1_2, ZERO_R1))) && (mtrx[0] == mtrx[1]) &&
        (mtrx[0] == mtrx[2]) && (mtrx[2] == -mtrx[3])) {
        H(target);
        return;
    }
    if (!freezeBasisH && (randGlobalPhase || (mtrx[0] == complex(M_SQRT1_2, ZERO_R1))) && (mtrx[0] == mtrx[1]) &&
        (mtrx[2] == -mtrx[3]) && (I_CMPLX * mtrx[0] == mtrx[2])) {
        H(target);
        S(target);
        return;
    }
    if (!freezeBasisH && (randGlobalPhase || (mtrx[0] == complex(M_SQRT1_2, ZERO_R1))) && (mtrx[0] == mtrx[2]) &&
        (mtrx[1] == -mtrx[3]) && (I_CMPLX * mtrx[2] == mtrx[3])) {
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

    if (shard.unit) {
        shard.unit->ApplySingleBit(trnsMtrx, shard.mapped);
    }
    if (DIRTY(shard)) {
        shard.MakeDirty();
        return;
    }

    complex Y0 = shard.amp0;
    shard.amp0 = (trnsMtrx[0] * Y0) + (trnsMtrx[1] * shard.amp1);
    shard.amp1 = (trnsMtrx[2] * Y0) + (trnsMtrx[3] * shard.amp1);
    if (doNormalize) {
        shard.ClampAmps(amplitudeFloor);
    }
}

void QUnit::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (IsIdentity(mtrx, true)) {
        return;
    }

    if (!norm(mtrx[1]) && !norm(mtrx[2])) {
        ApplyControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
        return;
    }

    if (!norm(mtrx[0]) && !norm(mtrx[3])) {
        ApplyControlledSingleInvert(controls, controlLen, target, mtrx[1], mtrx[2]);
        return;
    }

    CTRLED_GEN_WRAP(ApplyControlledSingleBit(CTRL_GEN_ARGS), ApplySingleBit(mtrx, target), false);
}

void QUnit::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (IsIdentity(mtrx, true)) {
        return;
    }

    if (!norm(mtrx[1]) && !norm(mtrx[2])) {
        ApplyAntiControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
        return;
    }

    if (!norm(mtrx[0]) && !norm(mtrx[3])) {
        ApplyAntiControlledSingleInvert(controls, controlLen, target, mtrx[1], mtrx[2]);
        return;
    }

    CTRLED_GEN_WRAP(ApplyAntiControlledSingleBit(CTRL_GEN_ARGS), ApplySingleBit(mtrx, target), true);
}

void QUnit::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CTRLED_SWAP_WRAP(CSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), false);
}

void QUnit::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSwap(CTRL_S_ARGS), Swap(qubit1, qubit2), true);
}

void QUnit::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CTRLED_SWAP_WRAP(CSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CTRLED_SWAP_WRAP(AntiCSqrtSwap(CTRL_S_ARGS), SqrtSwap(qubit1, qubit2), true);
}

void QUnit::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CTRLED_SWAP_WRAP(CISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), false);
}

void QUnit::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CTRLED_SWAP_WRAP(AntiCISqrtSwap(CTRL_S_ARGS), ISqrtSwap(qubit1, qubit2), true);
}

template <typename CF, typename F>
void QUnit::ApplyEitherControlled(const bitLenInt* controls, const bitLenInt& controlLen,
    const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F fn, const bool& inCurrentBasis)
{
    bitLenInt i;

    // If the controls start entirely separated from the targets, it's probably worth checking to see if the have
    // total or no probability of altering the targets, such that we can still keep them separate.

    std::vector<bitLenInt> controlVec;

    for (i = 0; i < controlLen; i++) {
        if (!inCurrentBasis) {
            RevertBasis1Qb(controls[i]);
            RevertBasis2Qb(controls[i], ONLY_INVERT, ONLY_TARGETS);
        }
        // If the shard's probability is cached, then it's free to check it, so we advance the loop.
        bool isEigenstate = false;
        if (shards[controls[i]].unit && shards[controls[i]].unit->isClifford()) {
            ProbBase(controls[i]);
        }
        if (!shards[controls[i]].isProbDirty) {
            // This might determine that we can just skip out of the whole gate, in which case it returns this
            // method:
            QEngineShard& shard = shards[controls[i]];
            if (IS_NORM_0(shard.amp1)) {
                if (!inCurrentBasis) {
                    Flush0Eigenstate(controls[i]);
                }
                if (!anti) {
                    /* This gate does nothing, so return without applying anything. */
                    return;
                }
                /* This control has 100% chance to "fire," so don't entangle it. */
                isEigenstate = true;
            } else if (IS_NORM_0(shard.amp0)) {
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

    for (i = 0; i < targets.size(); i++) {
        RevertBasis2Qb(targets[i]);
    }

    // TODO: If controls that survive the "first order" check above start out entangled,
    // then it might be worth checking whether there is any intra-unit chance of control bits
    // being conditionally all 0 or all 1, in any unit, due to entanglement.

    // If we've made it this far, we have to form the entangled representation and apply the gate.
    std::vector<bitLenInt> allBits(controlVec.size() + targets.size());
    std::copy(controlVec.begin(), controlVec.end(), allBits.begin());
    std::copy(targets.begin(), targets.end(), allBits.begin() + controlVec.size());
    // (Incidentally, we sort for the efficiency of QUnit's limited "mapper," a 1 dimensional array of qubits
    // without nearest neighbor restriction.)
    std::sort(allBits.begin(), allBits.end());

    std::vector<bitLenInt*> ebits(allBits.size());
    for (i = 0; i < allBits.size(); i++) {
        ebits[i] = &allBits[i];
    }

    QInterfacePtr unit = EntangleInCurrentBasis(ebits.begin(), ebits.end());

    std::vector<bitLenInt> controlsMapped(controlVec.size());
    for (i = 0; i < controlVec.size(); i++) {
        QEngineShard& cShard = shards[controlVec[i]];
        controlsMapped[i] = cShard.mapped;
        cShard.isPhaseDirty = true;
    }

    // This is the original method with the maximum number of non-entangled controls excised, (potentially leaving a
    // target bit in X or Y basis and acting as if Z basis by commutation).
    cfn(unit, controlsMapped);

    for (i = 0; i < targets.size(); i++) {
        shards[targets[i]].MakeDirty();
    }

    if (unit && unit->isClifford()) {
        for (i = 0; i < allBits.size(); i++) {
            ProbBase(allBits[i]);
        }
    }
}

bool QUnit::CArithmeticOptimize(bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>* controlVec)
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
        real1 prob = Prob(controls[i]);
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

void QUnit::CINC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (CArithmeticOptimize(controls, controlLen, &controlVec)) {
        // We've determined we can skip the entire gate.
        return;
    }

    // All cached classical control bits have been removed from controlVec.
    bitLenInt* lControls = new bitLenInt[controlVec.size()];
    std::copy(controlVec.begin(), controlVec.end(), lControls);
    DirtyShardIndexVector(controlVec);

    INT(toMod, start, length, 0xFF, false, lControls, controlVec.size());

    delete[] lControls;
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

    bool carry = (carryIndex < 0xFF);
    bool carryIn = carry && M(carryIndex);
    if (carry && (carryIn == isAdd)) {
        toMod++;
    }

    bitCapInt lengthPower = pow2(length);
    bitCapInt signMask = pow2(length - 1U);
    bitCapInt inOutInt = GetCachedPermutation(start, length);
    bitCapInt inInt = toMod;

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
    bitLenInt* controls, bitLenInt controlLen)
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

    std::vector<bitLenInt> allBits(controlLen + 1);
    std::copy(controls, controls + controlLen, allBits.begin());
    std::sort(allBits.begin(), allBits.begin() + controlLen);

    std::vector<bitLenInt*> ebits(allBits.size());
    for (bitLenInt i = 0; i < (ebits.size() - 1U); i++) {
        ebits[i] = &allBits[i];
    }

    bitLenInt* lControls = new bitLenInt[controlLen];

    // Try ripple addition, to avoid entanglement.
    bool toAdd, inReg;
    bool carry = false;
    int total;
    bitLenInt origLength = length;
    bitLenInt i = 0;
    while (i < origLength) {
        toAdd = (toMod & ONE_BCI) != 0;

        if (toAdd == carry) {
            toMod >>= ONE_BCI;
            start++;
            length--;
            i++;
            // Nothing is changed, in this bit. (The carry gets promoted to the next bit.)
            continue;
        }

        if (CheckBitPermutation(start)) {
            inReg = SHARD_STATE(shards[start]);
            total = (toAdd ? 1 : 0) + (inReg ? 1 : 0) + (carry ? 1 : 0);
            if (inReg != (total & 1)) {
                if (controlLen == 1U) {
                    CNOT(controls[0], start);
                } else if (controlLen) {
                    ApplyControlledSingleInvert(controls, controlLen, start, ONE_CMPLX, ONE_CMPLX);
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

                inReg = SHARD_STATE(shards[partStart]);
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
                    unit->CINC(partMod, shards[start].mapped, partLength, lControls, controlLen);
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
                ApplyControlledSingleInvert(controls, controlLen, carryIndex, ONE_CMPLX, ONE_CMPLX);
            } else {
                X(carryIndex);
            }
        }
        delete[] lControls;
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
            unit->CINC(toMod, shards[start].mapped, length, lControls, controlLen);
        } else {
            shards[start].unit->INC(toMod, shards[start].mapped, length);
        }
    }
    delete[] lControls;
}

void QUnit::INC(bitCapInt toMod, bitLenInt start, bitLenInt length) { INT(toMod, start, length, 0xFF, false); }

/// Add integer (without sign, with carry)
void QUnit::INCC(bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    if (M(carryIndex)) {
        X(carryIndex);
        toAdd++;
    }

    INT(toAdd, inOutStart, length, carryIndex, true);
}

/// Subtract integer (without sign, with carry)
void QUnit::DECC(bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
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

    bitLenInt signBit = start + length - 1U;
    bool knewFlagSet = CheckBitPermutation(overflowIndex);
    bool flagSet = SHARD_STATE(shards[overflowIndex]);

    if (knewFlagSet && !flagSet) {
        // Overflow detection is disabled
        INT(toMod, start, length, carryIndex, hasCarry);
        return;
    }

    bool addendNeg = (toMod & pow2(length - 1U)) != 0;
    bool knewSign = CheckBitPermutation(signBit);
    bool quantumNeg = SHARD_STATE(shards[signBit]);

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
        bitCapInt lengthMask = pow2Mask(length);
        bitCapInt res = GetCachedPermutation(inOutStart, length) * toMul;
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
        bitCapInt lengthMask = pow2Mask(length);
        bitCapInt origRes =
            GetCachedPermutation(inOutStart, length) | (GetCachedPermutation(carryStart, length) << length);
        bitCapInt res = origRes / toDiv;
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
        bitCapInt res = intPow(toMod, GetCachedPermutation(inStart, length)) % modN;
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
    for (bitLenInt i = 0; i < controlVec.size(); i++) {
        bits[i] = controlVec[i];
    }
    bits[controlVec.size()] = start;
    bits[controlVec.size() + 1] = carryStart;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(bits.size());
    for (bitLenInt i = 0; i < ebits.size(); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    if (controlVec.size()) {
        controlsMapped->resize(controlVec.size());
        for (bitLenInt i = 0; i < controlVec.size(); i++) {
            (*controlsMapped)[i] = shards[controlVec[i]].mapped;
            shards[controlVec[i]].isPhaseDirty = true;
        }
    }

    return unit;
}

void QUnit::CMULx(CMULFn fn, bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
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

void QUnit::CMUL(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        MUL(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QInterface::CMUL, toMod, start, carryStart, length, controls, controlLen);
}

void QUnit::CDIV(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        DIV(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QInterface::CDIV, toMod, start, carryStart, length, controls, controlLen);
}

void QUnit::CxMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen, bool inverse)
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
            bitLenInt* lControls = new bitLenInt[controlVec.size() + 1U];
            std::copy(controlVec.begin(), controlVec.end(), lControls);
            for (bitLenInt i = 0; i < length; i++) {
                lControls[controlVec.size()] = inStart + i;
                if (inverse) {
                    CDEC(toModExp, outStart, length, lControls, controlVec.size() + 1U);
                } else {
                    CINC(toModExp, outStart, length, lControls, controlVec.size() + 1U);
                }
                toModExp <<= ONE_BCI;
            }
            delete[] lControls;
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
    bitLenInt* controls, bitLenInt controlLen)
{
    CxMULModNOut(toMod, modN, inStart, outStart, length, controls, controlLen, false);
}

void QUnit::CIMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    CxMULModNOut(toMod, modN, inStart, outStart, length, controls, controlLen, true);
}

void QUnit::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
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
            if (DIRTY(shard)) {
                shard.MakeDirty();
                return;
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
        real1 prob = Prob(flagIndex);
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

bitCapInt QUnit::GetIndexedEigenstate(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(indexStart, indexLength);
    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt value = 0;
    for (bitCapIntOcl j = 0; j < valueBytes; j++) {
        value |= values[indexInt * valueBytes + j] << (8U * j);
    }

    return value;
}

bitCapInt QUnit::GetIndexedEigenstate(bitLenInt start, bitLenInt length, unsigned char* values)
{
    bitCapIntOcl indexInt = (bitCapIntOcl)GetCachedPermutation(start, length);
    bitLenInt bytes = (length + 7U) / 8U;
    bitCapInt value = 0;
    for (bitCapIntOcl j = 0; j < bytes; j++) {
        value |= values[indexInt * bytes + j] << (8U * j);
    }

    return value;
}

bitCapInt QUnit::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    unsigned char* values, bool resetValue)
{
    // TODO: Index bits that have exactly 0 or 1 probability can be optimized out of the gate.
    // This could follow the logic of UniformlyControlledSingleBit().
    // In the meantime, checking if all index bits are in eigenstates takes very little overhead.
    if (CheckBitsPermutation(indexStart, indexLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        SetReg(valueStart, valueLength, value);
#if ENABLE_VM6502Q_DEBUG
        return value;
#else
        return 0;
#endif
    }

    EntangleRange(indexStart, indexLength, valueStart, valueLength);

    bitCapInt toRet = shards[indexStart].unit->IndexedLDA(
        shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, values, resetValue);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);

    return toRet;
}

bitCapInt QUnit::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
#if ENABLE_VM6502Q_DEBUG
    if (CheckBitsPermutation(indexStart, indexLength) && CheckBitsPermutation(valueStart, valueLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        value = GetCachedPermutation(valueStart, valueLength) + value;
        bitCapInt valueMask = pow2Mask(valueLength);
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

    bitCapInt toRet = shards[indexStart].unit->IndexedADC(shards[indexStart].mapped, indexLength,
        shards[valueStart].mapped, valueLength, shards[carryIndex].mapped, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].MakeDirty();

    return toRet;
}

bitCapInt QUnit::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
#if ENABLE_VM6502Q_DEBUG
    if (CheckBitsPermutation(indexStart, indexLength) && CheckBitsPermutation(valueStart, valueLength)) {
        bitCapInt value = GetIndexedEigenstate(indexStart, indexLength, valueStart, valueLength, values);
        value = GetCachedPermutation(valueStart, valueLength) - value;
        bitCapInt valueMask = pow2Mask(valueLength);
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

    bitCapInt toRet = shards[indexStart].unit->IndexedSBC(shards[indexStart].mapped, indexLength,
        shards[valueStart].mapped, valueLength, shards[carryIndex].mapped, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].MakeDirty();

    return toRet;
}

void QUnit::Hash(bitLenInt start, bitLenInt length, unsigned char* values)
{
    if (CheckBitsPlus(start, length)) {
        // This operation happens to do nothing.
        return;
    }

    if (CheckBitsPermutation(start, length)) {
        bitCapInt value = GetIndexedEigenstate(start, length, values);
        SetReg(start, length, value);
        return;
    }

    DirtyShardRange(start, length);
    EntangleRange(start, length);
    shards[start].unit->Hash(shards[start].mapped, length, values);
}

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
    EndAllEmulation();
    ParallelUnitApply(
        [](QInterfacePtr unit, real1_f norm_thresh, real1_f unused2, int32_t unused3) {
            unit->UpdateRunningNorm(norm_thresh);
            return true;
        },
        norm_thresh);
}

void QUnit::NormalizeState(real1_f nrm, real1_f norm_thresh)
{
    EndAllEmulation();
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

void QUnit::Dump()
{
    ParallelUnitApply([](QInterfacePtr unit, real1_f unused1, real1_f unused2, int32_t unused3) {
        unit.reset();
        return true;
    });
}

bool QUnit::isFinished()
{
    return ParallelUnitApply(
        [](QInterfacePtr unit, real1_f unused1, real1_f unused2, int32_t unused3) { return unit->isFinished(); });
}

real1_f QUnit::SumSqrDiff(QUnitPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return 4.0f;
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
        return 4.0f;
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
    ToPermBasisAll();
    EndAllEmulation();

    QUnitPtr copyPtr = std::make_shared<QUnit>(
        engine, subEngine, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase, useHostRam);

    return CloneBody(copyPtr);
}

QInterfacePtr QUnit::CloneBody(QUnitPtr copyPtr)
{
    std::vector<QInterfacePtr> shardEngines;
    std::vector<QInterfacePtr> dupeEngines;
    std::vector<QInterfacePtr>::iterator origEngine;
    bitLenInt engineIndex;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (find(shardEngines.begin(), shardEngines.end(), shards[i].unit) == shardEngines.end()) {
            shardEngines.push_back(shards[i].unit);
            dupeEngines.push_back(shards[i].unit->Clone());
        }

        origEngine = find(shardEngines.begin(), shardEngines.end(), shards[i].unit);
        engineIndex = origEngine - shardEngines.begin();

        copyPtr->shards[i] = QEngineShard(shards[i]);
        copyPtr->shards[i].unit = dupeEngines[engineIndex];
    }

    return copyPtr;
}

void QUnit::ApplyBuffer(PhaseShardPtr phaseShard, const bitLenInt& control, const bitLenInt& target, const bool& isAnti)
{
    const bitLenInt controls[1] = { control };

    complex polarDiff = phaseShard->cmplxDiff;
    complex polarSame = phaseShard->cmplxSame;

    freezeBasis2Qb = true;
    if (phaseShard->isInvert) {
        if (isAnti) {
            ApplyAntiControlledSingleInvert(controls, 1U, target, polarSame, polarDiff);
        } else {
            ApplyControlledSingleInvert(controls, 1U, target, polarDiff, polarSame);
        }
    } else {
        if (isAnti) {
            ApplyAntiControlledSinglePhase(controls, 1U, target, polarSame, polarDiff);
        } else {
            ApplyControlledSinglePhase(controls, 1U, target, polarDiff, polarSame);
        }
    }
    freezeBasis2Qb = false;
}

void QUnit::ApplyBufferMap(const bitLenInt& bitIndex, ShardToPhaseMap bufferMap, const RevertExclusivity& exclusivity,
    const bool& isControl, const bool& isAnti, std::set<bitLenInt> exceptPartners, const bool& dumpSkipped)
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
                shard.RemovePhaseTarget(partner);
            }
            continue;
        }

        bitLenInt partnerIndex = FindShardIndex(partner);

        if (exceptPartners.find(partnerIndex) != exceptPartners.end()) {
            bufferMap.erase(phaseShard);
            if (dumpSkipped) {
                if (isControl) {
                    if (isAnti) {
                        shard.RemovePhaseAntiTarget(partner);
                    } else {
                        shard.RemovePhaseTarget(partner);
                    }
                } else {
                    if (isAnti) {
                        shard.RemovePhaseAntiControl(partner);
                    } else {
                        shard.RemovePhaseControl(partner);
                    }
                }
            }
            continue;
        }

        if (isControl) {
            if (isAnti) {
                shard.RemovePhaseAntiTarget(partner);
            } else {
                shard.RemovePhaseTarget(partner);
            }
            ApplyBuffer(phaseShard->second, bitIndex, partnerIndex, isAnti);
        } else {
            if (isAnti) {
                shard.RemovePhaseAntiControl(partner);
            } else {
                shard.RemovePhaseControl(partner);
            }
            ApplyBuffer(phaseShard->second, partnerIndex, bitIndex, isAnti);
        }

        bufferMap.erase(phaseShard);
    }
}

void QUnit::RevertBasis2Qb(const bitLenInt& i, const RevertExclusivity& exclusivity,
    const RevertControl& controlExclusivity, const RevertAnti& antiExclusivity, std::set<bitLenInt> exceptControlling,
    std::set<bitLenInt> exceptTargetedBy, const bool& dumpSkipped, const bool& skipOptimize)
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

void QUnit::CommuteH(const bitLenInt& bitIndex)
{
    QEngineShard& shard = shards[bitIndex];

    if (!QUEUED_PHASE(shard)) {
        return;
    }

    complex polarDiff, polarSame;
    ShardToPhaseMap::iterator phaseShard;
    QEngineShardPtr partner;
    PhaseShardPtr buffer;
    bitLenInt control;

    ShardToPhaseMap controlsShards = shard.controlsShards;

    for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
        buffer = phaseShard->second;
        partner = phaseShard->first;

        polarDiff = buffer->cmplxDiff;
        polarSame = buffer->cmplxSame;

        if (partner->isPauliX || partner->isPauliY || buffer->isInvert) {
            continue;
        }

        if (IS_ARG_0(polarDiff) && IS_ARG_PI(polarSame)) {
            shard.RemovePhaseTarget(partner);
            shard.AddPhaseAngles(partner, ONE_CMPLX, -ONE_CMPLX);
        } else if (IS_ARG_PI(polarDiff) && IS_ARG_0(polarSame)) {
            shard.RemovePhaseTarget(partner);
            shard.AddAntiPhaseAngles(partner, -ONE_CMPLX, ONE_CMPLX);
        }
    }

    controlsShards = shard.antiControlsShards;

    for (phaseShard = controlsShards.begin(); phaseShard != controlsShards.end(); phaseShard++) {
        buffer = phaseShard->second;
        partner = phaseShard->first;

        polarDiff = buffer->cmplxDiff;
        polarSame = buffer->cmplxSame;

        if (partner->isPauliX || partner->isPauliY || buffer->isInvert) {
            continue;
        }

        if (IS_ARG_0(polarDiff) && IS_ARG_PI(polarSame)) {
            shard.RemovePhaseAntiTarget(partner);
            shard.AddAntiPhaseAngles(partner, ONE_CMPLX, -ONE_CMPLX);
        } else if (IS_ARG_PI(polarDiff) && IS_ARG_0(polarSame)) {
            shard.RemovePhaseAntiTarget(partner);
            shard.AddPhaseAngles(partner, -ONE_CMPLX, ONE_CMPLX);
        }
    }

    RevertBasis2Qb(bitIndex, INVERT_AND_PHASE, ONLY_CONTROLS, CTRL_AND_ANTI, {}, {}, false, true);

    if (!QUEUED_PHASE(shard)) {
        return;
    }

    bool isSame, isOpposite;

    ShardToPhaseMap targetOfShards = shard.targetOfShards;

    for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
        buffer = phaseShard->second;

        polarDiff = buffer->cmplxDiff;
        polarSame = buffer->cmplxSame;

        partner = phaseShard->first;

        // If isSame and !isInvert, application of this buffer is already "efficient."
        isSame = (buffer->isInvert || (!partner->isPauliX && !partner->isPauliY) || !partner->IsInvertTarget()) &&
            IS_SAME(polarDiff, polarSame);
        isOpposite = !buffer->isInvert && IS_OPPOSITE(polarDiff, polarSame);

        if (isSame || isOpposite) {
            continue;
        }

        control = FindShardIndex(partner);
        shard.RemovePhaseControl(partner);
        ApplyBuffer(buffer, control, bitIndex, false);
    }

    targetOfShards = shard.antiTargetOfShards;

    for (phaseShard = targetOfShards.begin(); phaseShard != targetOfShards.end(); phaseShard++) {
        buffer = phaseShard->second;

        polarDiff = buffer->cmplxDiff;
        polarSame = buffer->cmplxSame;

        partner = phaseShard->first;

        // If isSame and !isInvert, application of this buffer is already "efficient."
        isSame = (buffer->isInvert || (!partner->isPauliX && !partner->isPauliY) || !partner->IsInvertTarget()) &&
            IS_SAME(polarDiff, polarSame);
        isOpposite = !buffer->isInvert && IS_OPPOSITE(polarDiff, polarSame);

        if (isSame || isOpposite) {
            continue;
        }

        control = FindShardIndex(partner);
        shard.RemovePhaseAntiControl(partner);
        ApplyBuffer(buffer, control, bitIndex, true);
    }

    shard.CommuteH();
}

void QUnit::OptimizePairBuffers(const bitLenInt& control, const bitLenInt& target, const bool& anti)
{
    QEngineShard& cShard = shards[control];
    QEngineShard& tShard = shards[target];

    ShardToPhaseMap& targets = anti ? tShard.antiTargetOfShards : tShard.targetOfShards;

    ShardToPhaseMap::iterator phaseShard = targets.find(&cShard);

    if ((phaseShard == targets.end()) || phaseShard->second->isInvert) {
        return;
    }

    PhaseShardPtr buffer = phaseShard->second;

    if (IS_NORM_0(buffer->cmplxDiff - buffer->cmplxSame)) {
        tShard.RemovePhaseControl(&cShard);
        ApplyBuffer(buffer, control, target, anti);
        return;
    }

    ShardToPhaseMap& antiTargets = anti ? tShard.targetOfShards : tShard.antiTargetOfShards;

    ShardToPhaseMap::iterator antiShard = tShard.antiTargetOfShards.find(&cShard);

    if ((antiShard == antiTargets.end()) || antiShard->second->isInvert) {
        return;
    }

    PhaseShardPtr aBuffer = antiShard->second;

    if (IS_NORM_0(buffer->cmplxDiff - aBuffer->cmplxSame) && IS_NORM_0(buffer->cmplxSame - aBuffer->cmplxDiff)) {
        tShard.RemovePhaseControl(&cShard);
        tShard.RemovePhaseAntiControl(&cShard);
        if (anti) {
            ApplySinglePhase(buffer->cmplxSame, buffer->cmplxDiff, target);
        } else {
            ApplySinglePhase(buffer->cmplxDiff, buffer->cmplxSame, target);
        }
    }
}

} // namespace Qrack
