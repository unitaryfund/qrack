//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
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

#define SHARD_STATE(shard) (norm(shard.amp0) < (ONE_R1 / 2))
#define QUEUED_PHASE(shard) ((shard.targetOfShards.size() != 0) || (shard.controlsShards.size() != 0))
#define QUEUED_H_PHASE(shard) (shard.isPlusMinus && QUEUED_PHASE(shard))
/* "UNSAFE" variants here do not check whether the bit is in |0>/|1> rather than |+>/|-> basis. */
#define UNSAFE_CACHED_CLASSICAL(shardIndex)                                                                            \
    (!shards[shardIndex].isProbDirty &&                                                                                \
        ((ProbBase(shardIndex) < min_norm) || ((ONE_R1 - ProbBase(shardIndex)) < min_norm)))
#define CACHED_1QB(shardIndex) (!shards[shardIndex].isProbDirty && !shards[shardIndex].isPlusMinus)
#define CACHED_1QB_H(shardIndex)                                                                                       \
    (shards[shardIndex].isPlusMinus && !QUEUED_PHASE(shards[shardIndex]) && UNSAFE_CACHED_CLASSICAL(shardIndex))
#define CACHED_PROB(shardIndex)                                                                                        \
    (CACHED_1QB(shardIndex) && (shards[shardIndex].targetOfShards.size() == 0) &&                                      \
        (shards[shardIndex].controlsShards.size() == 0))
#define CACHED_CLASSICAL(shardIndex)                                                                                   \
    (CACHED_PROB(shardIndex) && ((Prob(shardIndex) < min_norm) || ((ONE_R1 - Prob(shardIndex)) < min_norm)))
#define CACHED_ONE(shardIndex) (CACHED_PROB(shardIndex) && ((ONE_R1 - Prob(shardIndex)) < min_norm))
#define CACHED_ZERO(shardIndex) (CACHED_PROB(shardIndex) && (Prob(shardIndex) < min_norm))
#define PHASE_MATTERS(shardIndex) (!randGlobalPhase || !CACHED_CLASSICAL(shardIndex))
#define DIRTY(shard) (shard.isPhaseDirty || shard.isProbDirty)
#define IS_ONE_CMPLX(c) (c == ONE_CMPLX)

namespace Qrack {

QUnit::QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<bitLenInt> devList)
    : QUnit(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceID,
          useHardwareRNG, useSparseStateVec, norm_thresh, devList)
{
    // Intentionally left blank
}

QUnit::QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID,
    bool useHardwareRNG, bool useSparseStateVec, real1 norm_thresh, std::vector<bitLenInt> devList)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , engine(eng)
    , subengine(subEng)
    , devID(deviceID)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , freezeBasis(false)
{
    shards.resize(qBitCount);

    bool bitState;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((initState >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards[i] = QEngineShard(MakeEngine(1, bitState ? 1 : 0), bitState);
    }
}

QInterfacePtr QUnit::MakeEngine(bitLenInt length, bitCapInt perm)
{
    return CreateQuantumInterface(engine, subengine, length, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse);
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool bitState;

    Finish();

    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((perm >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards[i] = QEngineShard(MakeEngine(1, bitState ? 1 : 0), bitState);
    }
}

void QUnit::SetQuantumState(const complex* inputState)
{
    QInterfacePtr unit = MakeEngine(qubitCount, 0);
    unit->SetQuantumState(inputState);

    for (bitLenInt idx = 0; idx < qubitCount; idx++) {
        shards[idx] = QEngineShard(unit, idx);
    }

    if (qubitCount == 1U) {
        QEngineShard& shard = shards[0];
        shard.isEmulated = false;
        shard.isProbDirty = false;
        shard.isPhaseDirty = false;
        shard.amp0 = inputState[0];
        shard.amp1 = inputState[1];
        shard.isPlusMinus = false;
    }
}

void QUnit::GetQuantumState(complex* outputState)
{
    ToPermBasisAll();
    EndAllEmulation();

    QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(Clone());
    clone->OrderContiguous(clone->EntangleAll());
    clone->shards[0].unit->GetQuantumState(outputState);
}

void QUnit::GetProbs(real1* outputProbs)
{
    ToPermBasisAll();
    EndAllEmulation();

    QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(Clone());
    clone->OrderContiguous(clone->EntangleAll());
    clone->shards[0].unit->GetProbs(outputProbs);
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
        if (norm(result) == ZERO_R1) {
            break;
        }
    }

    if ((shards[0].unit->GetQubitCount() > 1) && (norm(result) == ONE_R1) &&
        (randGlobalPhase || (result == ONE_CMPLX))) {
        SetPermutation(perm);
    }

    return result;
}

void QUnit::SetAmplitude(bitCapInt perm, complex amp)
{
    EntangleAll();
    shards[0].unit->SetAmplitude(perm, amp);
}

bitLenInt QUnit::Compose(QUnitPtr toCopy, bool isConsumed) { return Compose(toCopy, qubitCount, isConsumed); }

/*
 * Append QInterface in the middle of QUnit.
 */
bitLenInt QUnit::Compose(QUnitPtr toCopy, bitLenInt start, bool isConsumed)
{
    /* Create a clone of the quantum state in toCopy. */
    QUnitPtr clone = std::dynamic_pointer_cast<QUnit>(toCopy->Clone());

    /* Insert the new shards in the middle */
    shards.insert(shards.begin() + start, clone->shards.begin(), clone->shards.end());

    SetQubitCount(qubitCount + toCopy->GetQubitCount());

    return start;
}

void QUnit::Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
{
    /* TODO: This method should compose the bits for the destination without cohering the length first */

    for (bitLenInt i = 0; i < length; i++) {
        RevertBasis2Qb(start + i);
    }

    QInterfacePtr destEngine;
    if (length > 1) {
        EntangleRange(start, length);
        OrderContiguous(shards[start].unit);

        if (dest) {
            dest->EntangleRange(0, length);
            dest->OrderContiguous(dest->shards[0].unit);

            destEngine = dest->shards[0].unit;
        }
    }

    QInterfacePtr unit = shards[start].unit;
    bitLenInt mapped = shards[start].mapped;
    bitLenInt unitLength = unit->GetQubitCount();

    if (dest) {
        for (bitLenInt i = 0; i < length; i++) {
            QInterfacePtr tempUnit = dest->shards[start + i].unit;
            dest->shards[start + i] = QEngineShard(shards[start + i]);
            dest->shards[start + i].unit = tempUnit;
        }

        unit->Decompose(mapped, length, destEngine);
    } else {
        unit->Dispose(mapped, length);
    }

    unit->Finish();
    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);

    if (unitLength == length) {
        return;
    }

    /* Find the rest of the qubits. */
    for (auto&& shard : shards) {
        if (shard.unit == unit && shard.mapped >= (mapped + length)) {
            shard.mapped -= length;
        }
    }
}

void QUnit::Decompose(bitLenInt start, bitLenInt length, QUnitPtr dest) { Detach(start, length, dest); }

void QUnit::Dispose(bitLenInt start, bitLenInt length) { Detach(start, length, nullptr); }

// The optimization of this method is redudandant with other optimizations in QUnit.
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
            bitLenInt offset = unit1->Compose(consumed, true);
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
            offsets[consumed] = retained->Compose(consumed, true);
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

/*
 * Accept a variable number of bits, entangle them all into a single QInterface
 * object, and then call the supplied function on that object.
 */
template <typename F, typename... B> void QUnit::EntangleAndCallMember(F fn, B... bits)
{
    auto qbits = Entangle({ &bits... });
    ((*qbits).*fn)(bits...);
}

template <typename F, typename... B> void QUnit::EntangleAndCall(F fn, B... bits)
{
    auto qbits = Entangle({ &bits... });
    fn(qbits, bits...);
}

template <typename F, typename... B> void QUnit::EntangleAndCallMemberRot(F fn, real1 radians, B... bits)
{
    auto qbits = Entangle({ &bits... });
    ((*qbits).*fn)(radians, bits...);
}

bool QUnit::TrySeparate(bitLenInt start, bitLenInt length)
{
    if (length == qubitCount) {
        return true;
    }

    if ((length == 1) && (shards[start].unit->GetQubitCount() == 1)) {
        return true;
    }

    if (length > 1) {
        EntangleRange(start, length);
        OrderContiguous(shards[start].unit);
    } else {
        // If length == 1, this is usually all that's worth trying:
        QEngineShard& shard = shards[start];
        real1 prob;
        if (shard.isPlusMinus || QUEUED_PHASE(shard)) {
            prob = ProbBase(start);
        } else {
            prob = Prob(start);
        }
        return ((prob < min_norm) || ((ONE_R1 - prob) < min_norm));
    }

    QInterfacePtr separatedBits = MakeEngine(length, 0);

    QInterfacePtr unitCopy = shards[start].unit->Clone();

    bitLenInt mappedStart = shards[start].mapped;
    unitCopy->Decompose(mappedStart, length, separatedBits);
    unitCopy->Compose(separatedBits, mappedStart);

    bool didSeparate = unitCopy->ApproxCompare(shards[start].unit);
    if (didSeparate) {
        // The subsystem is separable.
        shards[start].unit->Dispose(mappedStart, length);

        /* Find the rest of the qubits. */
        for (auto&& shard : shards) {
            if (shard.unit == shards[start].unit && shard.mapped >= (mappedStart + length)) {
                shard.mapped -= length;
            }
        }

        for (bitLenInt i = 0; i < length; i++) {
            shards[start + i].unit = separatedBits;
            shards[start + i].mapped = i;
        }
    }

    return didSeparate;
}

void QUnit::OrderContiguous(QInterfacePtr unit)
{
    /* Before we call OrderContinguous, when we are cohering lists of shards, we should always proactively sort the
     * order in which we compose qubits into a single engine. This is a cheap way to reduce the need for costly qubit
     * swap gates, later. */

    if (unit->GetQubitCount() == 1) {
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
    if (UNSAFE_CACHED_CLASSICAL(qubitIndex)) {
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
        if (!(CACHED_1QB_H(qubitIndex + i) && (ProbBase(qubitIndex + i) < min_norm))) {
            isHBasis = false;
            break;
        }
    }

    return isHBasis;
}

void QUnit::DumpShards()
{
    int i = 0;
    for (auto shard : shards) {
        printf("%2d.\t%p[%d]\n", i++, shard.unit.get(), (int)shard.mapped);
    }
}

real1 QUnit::ProbBase(const bitLenInt& qubit)
{
    QEngineShard& shard = shards[qubit];

    if (shard.isProbDirty) {
        real1 prob = (shard.unit->Prob)(shard.mapped);
        shard.amp1 = complex(sqrt(prob), ZERO_R1);
        shard.amp0 = complex(sqrt(ONE_R1 - prob), ZERO_R1);
        shard.isProbDirty = false;

        if (norm(shard.amp0) < min_norm) {
            SeparateBit(true, qubit);
        } else if (norm(shard.amp1) < min_norm) {
            SeparateBit(false, qubit);
        }
    }

    return norm(shard.amp1);
}

real1 QUnit::Prob(bitLenInt qubit)
{
    ToPermBasis(qubit);
    return ProbBase(qubit);
}

real1 QUnit::ProbAll(bitCapInt perm) { return clampProb(norm(GetAmplitude(perm))); }

void QUnit::SeparateBit(bool value, bitLenInt qubit, bool doDispose)
{
    RevertBasis2Qb(qubit);

    QInterfacePtr unit = shards[qubit].unit;
    bitLenInt mapped = shards[qubit].mapped;

    shards[qubit].unit = MakeEngine(1, value ? 1 : 0);
    shards[qubit].mapped = 0;
    shards[qubit].isEmulated = false;
    shards[qubit].isProbDirty = false;
    shards[qubit].isPhaseDirty = false;
    shards[qubit].amp0 = value ? complex(ZERO_R1, ZERO_R1) : complex(ONE_R1, ZERO_R1);
    shards[qubit].amp1 = value ? complex(ONE_R1, ZERO_R1) : complex(ZERO_R1, ZERO_R1);

    if (unit->GetQubitCount() == 1) {
        return;
    }

    if (doDispose) {
        unit->Dispose(mapped, 1, value ? ONE_BCI : 0);
    }

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
    if (CACHED_CLASSICAL(qubit)) {
        result = doForce ? res : SHARD_STATE(shard);
    } else if (!shard.isProbDirty && (shard.unit->GetQubitCount() == 1U)) {
        result = doForce ? res : (Rand() <= norm(shard.amp1));
    } else {
        EndEmulation(qubit);
        result = shard.unit->ForceM(shard.mapped, res, doForce, doApply);
    }

    if (!doApply) {
        return result;
    }

    if (shard.unit->GetQubitCount() == 1U) {
        shard.isProbDirty = false;
        shard.isPhaseDirty = false;
        shard.isEmulated = true;
        shard.amp0 = result ? ZERO_CMPLX : ONE_CMPLX;
        shard.amp1 = result ? ONE_CMPLX : ZERO_CMPLX;

        /* If we're keeping the bits, and they're already in their own unit, there's nothing to do. */
        return result;
    }

    // This is critical: it's the "nonlocal correlation" of "wave function collapse".
    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (shards[i].unit == shard.unit) {
            shards[i].isProbDirty = true;
            shards[i].isPhaseDirty = true;
        }
    }

    SeparateBit(result, qubit);

    return result;
}

bitCapInt QUnit::ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values, bool doApply)
{
    //"Collapsing" the register bit-by-bit is very costly. It's cheap to eventually recover the measurement from the
    // single-bit method, but only once we collapse the state more efficiently.

    bitLenInt i, j;
    bitLenInt contigRegStart;
    bitLenInt contigRegLength;
    bitCapInt contigRegResult;

    for (i = 0; i < length; i++) {
        ToPermBasis(bits[i]);
        EndEmulation(bits[i]);
    }

    std::map<QInterfacePtr, std::vector<bitLenInt>> perms;
    std::map<QInterfacePtr, std::map<bitLenInt, bitLenInt>> invMap;
    for (i = 0; i < length; i++) {
        QEngineShard& shard = shards[bits[i]];
        perms[shard.unit].push_back(shard.mapped);
        invMap[shard.unit][shard.mapped] = i;
    }

    for (auto&& qi : perms) {
        if (qi.second.size() == 1) {
            continue;
        }

        bool doApplySub = doApply;
        std::sort(qi.second.begin(), qi.second.end());
        i = 0;
        std::map<bitLenInt, bitLenInt> toDispose;
        std::vector<bitCapInt> partResults;
        while (i < qi.second.size()) {
            contigRegStart = qi.second[i];
            contigRegLength = 1U;
            i++;
            while (qi.second[i] == (contigRegStart + contigRegLength)) {
                contigRegLength++;
                i++;
            }
            // If we're measuring the entire length of the sub-unit, we can skip the "collapse" portion of measurement.
            doApplySub = doApply && (qi.first->GetQubitCount() != contigRegLength);
            toDispose[contigRegStart] = contigRegLength;
            if (values) {
                contigRegResult = 0;
                for (j = 0; j < contigRegLength; j++) {
                    if (values[invMap[qi.first][contigRegStart + j]]) {
                        contigRegResult |= pow2(j);
                    }
                }
                if (doApplySub) {
                    partResults.push_back(qi.first->ForceMReg(contigRegStart, contigRegLength, contigRegResult, true));
                } else {
                    partResults.push_back(contigRegResult);
                }
            } else {
                partResults.push_back(qi.first->ForceMReg(contigRegStart, contigRegLength, 0, false, doApplySub));
            }
        }

        // This is critical: it's the "nonlocal correlation" of "wave function collapse".
        if (doApplySub) {
            for (j = 0; j < qubitCount; j++) {
                if (shards[j].unit == qi.first) {
                    shards[j].isProbDirty = true;
                    shards[j].isPhaseDirty = true;
                }
            }
        }

        for (auto iter = toDispose.rbegin(); iter != toDispose.rend(); ++iter) {
            contigRegResult = partResults.back();
            for (i = iter->first; i < (iter->first + iter->second); i++) {
                SeparateBit((contigRegResult & ONE_BCI) != 0, bits[invMap[qi.first][i]], false);
                contigRegResult >>= ONE_BCI;
            }
            contigRegResult = partResults.back();
            partResults.pop_back();
            if (qi.first->GetQubitCount() > iter->second) {
                qi.first->Dispose(iter->first, iter->second, contigRegResult);
            }
        }
    }

    return QInterface::ForceM(bits, length, values, doApply);
}

bitCapInt QUnit::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    if (isSparse) {
        return QInterface::ForceMReg(start, length, result, doForce, doApply);
    }

    bitLenInt i;

    ToPermBasisMeasure(start, length);

    bitLenInt* bits = new bitLenInt[length];
    for (i = 0; i < length; i++) {
        bits[i] = start + i;
    }

    bool* values = NULL;
    if (doForce) {
        values = new bool[length];
        for (i = 0; i < length; i++) {
            values[i] = ((result >> (bitCapIntOcl)i) & 1U) != 0;
        }
    }

    bitCapInt maskResult = ForceM(bits, length, values, doApply);

    bitCapInt toRet = 0;
    for (i = 0; i < length; i++) {
        toRet |= (maskResult & pow2(start + i)) ? pow2(i) : 0;
    }

    delete[] bits;
    if (values) {
        delete[] values;
    }

    return toRet;
}

/// Set register bits to given permutation
void QUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    MReg(start, length);

    bool bitState;
    for (bitLenInt i = 0; i < length; i++) {
        bitState = ((value >> (bitCapIntOcl)i) & ONE_BCI) != 0;
        shards[i + start] = QEngineShard(shards[i + start].unit, bitState);
        shards[i + start].isEmulated = true;
    }
}

void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    // Simply swap the bit mapping, so long as no 2-qubit gates are buffered.
    std::swap(shard1, shard2);

    QInterfacePtr unit = shards[qubit1].unit;
    if (unit == shards[qubit2].unit) {
        OrderContiguous(unit);
    }
}

/* Unfortunately, many methods are overloaded, which prevents using just the address-to-member. */
#define PTR3(OP) (void (QInterface::*)(bitLenInt, bitLenInt, bitLenInt))(&QInterface::OP)
#define PTR2(OP) (void (QInterface::*)(bitLenInt, bitLenInt))(&QInterface::OP)
#define PTR1(OP) (void (QInterface::*)(bitLenInt))(&QInterface::OP)
#define PTR2A(OP) (void (QInterface::*)(real1, bitLenInt, bitLenInt))(&QInterface::OP)
#define PTRA(OP) (void (QInterface::*)(real1, bitLenInt))(&QInterface::OP)

void QUnit::ISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    RevertBasis2Qb(qubit1);
    RevertBasis2Qb(qubit2);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (CACHED_CLASSICAL(qubit1) && CACHED_CLASSICAL(qubit2)) {
        // We can avoid dirtying the cache and entangling, since the bits are classical.
        if (SHARD_STATE(shard1) != SHARD_STATE(shard2)) {
            Swap(qubit1, qubit2);
            if (!randGlobalPhase) {
                // Under the preconditions, this has no effect on Hermitian expectation values, but we track it, if the
                // QUnit is tracking arbitrary numerical phase.
                ApplySinglePhase(I_CMPLX, I_CMPLX, 0);
            }
        }
        return;
    }

    EntangleAndCallMember(PTR2(ISwap), qubit1, qubit2);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.isProbDirty = true;
    shard1.isPhaseDirty = true;
    shard2.isProbDirty = true;
    shard2.isPhaseDirty = true;
}

void QUnit::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    RevertBasis2Qb(qubit1);
    RevertBasis2Qb(qubit2);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (CACHED_CLASSICAL(qubit1) && CACHED_CLASSICAL(qubit2) && (SHARD_STATE(shard1) == SHARD_STATE(shard2))) {
        // We can avoid dirtying the cache and entangling, since this gate doesn't swap identical classical bits.
        return;
    }

    EntangleAndCallMember(PTR2(SqrtSwap), qubit1, qubit2);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.isProbDirty = true;
    shard1.isPhaseDirty = true;
    shard2.isProbDirty = true;
    shard2.isPhaseDirty = true;
}

void QUnit::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    RevertBasis2Qb(qubit1);
    RevertBasis2Qb(qubit2);

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    if (CACHED_CLASSICAL(qubit1) && CACHED_CLASSICAL(qubit2) && (SHARD_STATE(shard1) == SHARD_STATE(shard2))) {
        // We can avoid dirtying the cache and entangling, since this gate doesn't swap identical classical bits.
        return;
    }

    EntangleAndCallMember(PTR2(ISqrtSwap), qubit1, qubit2);

    // TODO: If we multiply out cached amplitudes, we can optimize this.

    shard1.isProbDirty = true;
    shard1.isPhaseDirty = true;
    shard2.isProbDirty = true;
    shard2.isPhaseDirty = true;
}

void QUnit::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    // If there are no controls, this is equivalent to the single bit gate.
    if (controlLen == 0) {
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
    if (trimmedControls.size() == 0) {
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

    shards[qubitIndex].isProbDirty = true;
    shards[qubitIndex].isPhaseDirty = true;

    delete[] mappedControls;
}

void QUnit::H(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (!freezeBasis) {
        if (!shard.TryHCommute()) {
            RevertBasis2Qb(target);
        }
        shard.isPlusMinus = !shard.isPlusMinus;
        return;
    }

    ApplyOrEmulate(shard, [&](QEngineShard& shard) { shard.unit->H(shard.mapped); });

    if (DIRTY(shard)) {
        shard.isProbDirty = true;
        shard.isPhaseDirty = true;
        return;
    }

    complex tempAmp1 = ((real1)M_SQRT1_2) * (shard.amp0 - shard.amp1);
    shard.amp0 = ((real1)M_SQRT1_2) * (shard.amp0 + shard.amp1);
    shard.amp1 = tempAmp1;
    shard.ClampAmps();

    if (shard.unit->GetQubitCount() > 1U) {
        if (norm(shard.amp0) < min_norm) {
            SeparateBit(true, target);
        } else if (norm(shard.amp1) < min_norm) {
            SeparateBit(false, target);
        }
    }
}

void QUnit::XBase(const bitLenInt& target)
{
    QEngineShard& shard = shards[target];
    ApplyOrEmulate(shard, [&](QEngineShard& shard) { shard.unit->X(shard.mapped); });
    std::swap(shard.amp0, shard.amp1);
}

void QUnit::ZBase(const bitLenInt& target)
{
    QEngineShard& shard = shards[target];
    // If the target bit is in a |0>/|1> eigenstate, this gate has no effect.
    if (PHASE_MATTERS(target)) {
        EndEmulation(shard);
        shard.unit->Z(shard.mapped);
        shard.amp1 = -shard.amp1;
    }
}

void QUnit::X(bitLenInt target)
{
    QEngineShard& shard = shards[target];

    RevertBasis2Qb(target, false, true);

    // Not necessary to check return after the above revert:
    shard.TryFlipPhaseAnti();

    if (!shard.isPlusMinus) {
        XBase(target);
    } else {
        ZBase(target);
    }
}

void QUnit::Z(bitLenInt target)
{
    // Commutes with controlled phase optimizations
    QEngineShard& shard = shards[target];

    RevertBasis2Qb(target, true);

    // Not necessary to check return after the above revert:
    shard.TryCommutePhase(ONE_CMPLX, -ONE_CMPLX);

    if (!shard.isPlusMinus) {
        if (PHASE_MATTERS(target)) {
            ApplyOrEmulate(shard, [&](QEngineShard& shard) { shard.unit->Z(shard.mapped); });
            shard.amp1 = -shard.amp1;
        }
    } else {
        XBase(target);
    }
}

void QUnit::Transform2x2(const complex* mtrxIn, complex* mtrxOut)
{
    mtrxOut[0] = (ONE_R1 / 2) * (mtrxIn[0] + mtrxIn[1] + mtrxIn[2] + mtrxIn[3]);
    mtrxOut[1] = (ONE_R1 / 2) * (mtrxIn[0] - mtrxIn[1] + mtrxIn[2] - mtrxIn[3]);
    mtrxOut[2] = (ONE_R1 / 2) * (mtrxIn[0] + mtrxIn[1] - mtrxIn[2] - mtrxIn[3]);
    mtrxOut[3] = (ONE_R1 / 2) * (mtrxIn[0] - mtrxIn[1] - mtrxIn[2] + mtrxIn[3]);
}

void QUnit::TransformPhase(const complex& topLeft, const complex& bottomRight, complex* mtrxOut)
{
    mtrxOut[0] = (ONE_R1 / 2) * (topLeft + bottomRight);
    mtrxOut[1] = (ONE_R1 / 2) * (topLeft - bottomRight);
    mtrxOut[2] = (ONE_R1 / 2) * (topLeft - bottomRight);
    mtrxOut[3] = (ONE_R1 / 2) * (topLeft + bottomRight);
}

void QUnit::TransformInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut)
{
    mtrxOut[0] = (ONE_R1 / 2) * (bottomLeft + topRight);
    mtrxOut[1] = (ONE_R1 / 2) * (-bottomLeft + topRight);
    mtrxOut[2] = (ONE_R1 / 2) * (bottomLeft - topRight);
    mtrxOut[3] = (ONE_R1 / 2) * -(bottomLeft + topRight);
}

#define CTRLED_GEN_WRAP(ctrld, bare, anti)                                                                             \
    ApplyEitherControlled(controls, controlLen, { target }, anti,                                                      \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {                                               \
            complex trnsMtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                                  \
            if (!shards[target].isPlusMinus) {                                                                         \
                std::copy(mtrx, mtrx + 4, trnsMtrx);                                                                   \
            } else {                                                                                                   \
                Transform2x2(mtrx, trnsMtrx);                                                                          \
            }                                                                                                          \
            unit->ctrld;                                                                                               \
        },                                                                                                             \
        [&]() { bare; });

#define CTRLED_PHASE_INVERT_WRAP(ctrld, ctrldgen, bare, anti, isInvert, top, bottom)                                   \
    ApplyEitherControlled(controls, controlLen, { target }, anti,                                                      \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {                                               \
            if (!shards[target].isPlusMinus) {                                                                         \
                unit->ctrld;                                                                                           \
            } else {                                                                                                   \
                complex trnsMtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };                              \
                if (isInvert) {                                                                                        \
                    TransformInvert(top, bottom, trnsMtrx);                                                            \
                } else {                                                                                               \
                    TransformPhase(top, bottom, trnsMtrx);                                                             \
                }                                                                                                      \
                unit->ctrldgen;                                                                                        \
            }                                                                                                          \
        },                                                                                                             \
        [&]() { bare; });

#define CTRLED_SWAP_WRAP(ctrld, bare, anti)                                                                            \
    if (qubit1 == qubit2) {                                                                                            \
        return;                                                                                                        \
    }                                                                                                                  \
    ToPermBasis(qubit1);                                                                                               \
    ToPermBasis(qubit2);                                                                                               \
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, anti,                                              \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->ctrld; }, [&]() { bare; })
#define CTRL_GEN_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, trnsMtrx
#define CTRL_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, mtrx
#define CTRL_1_ARGS mappedControls[0], shards[target].mapped
#define CTRL_2_ARGS mappedControls[0], mappedControls[1], shards[target].mapped
#define CTRL_S_ARGS &(mappedControls[0]), mappedControls.size(), shards[qubit1].mapped, shards[qubit2].mapped
#define CTRL_P_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, topLeft, bottomRight
#define CTRL_I_ARGS &(mappedControls[0]), mappedControls.size(), shards[target].mapped, topRight, bottomLeft

bool QUnit::TryCnotOptimize(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex& bottomLeft, const complex& topRight, const bool& anti)
{
    // Returns:
    // "true" if successfully handled/consumed,
    // "false" if needs general handling

    bitLenInt rControl = 0;
    bitLenInt rControlLen = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        QEngineShard& shard = shards[controls[i]];
        if (CACHED_CLASSICAL(controls[i])) {
            if ((anti && SHARD_STATE(shard)) || (!anti && !SHARD_STATE(shard))) {
                return true;
            }
        } else {
            rControl = controls[i];
            rControlLen++;
            if (rControlLen > 1U) {
                break;
            }
        }
    }

    if (rControlLen == 0U) {
        ApplySingleInvert(topRight, bottomLeft, target);
        return true;
    } else if (rControlLen == 1U) {
        complex iTest[4] = { topRight, 0, 0, bottomLeft };
        if (IsIdentity(iTest, true)) {
            if (anti) {
                AntiCNOT(rControl, target);
            } else {
                CNOT(rControl, target);
            }
            return true;
        }
    }

    return false;
}

void QUnit::CNOT(bitLenInt control, bitLenInt target)
{
    if (CACHED_PROB(control)) {
        if (Prob(control) < min_norm) {
            return;
        }
        if ((ONE_R1 - Prob(control)) < min_norm) {
            X(target);
            return;
        }
    }

    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;

    QEngineShard& cShard = shards[control];
    QEngineShard& tShard = shards[target];

    // We're free to transform gates to any orthonormal basis of the Hilbert space.
    // For a 2 qubit system, if the control is the lefthand bit, it's easy to verify the following truth table for CNOT:
    // |++> -> |++>
    // |+-> -> |-->
    // |-+> -> |-+>
    // |--> -> |+->
    // Under the Jacobian transformation between these two bases for defining the truth table, the matrix representation
    // is equivalent to the gate with bits flipped. We just let ApplyEitherControlled() know to leave the current basis
    // alone, by way of the last optional "true" argument in the call.
    if (cShard.isPlusMinus && tShard.isPlusMinus) {
        std::swap(controls[0], target);
        ApplyEitherControlled(controls, controlLen, { target }, false,
            [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->CNOT(CTRL_1_ARGS); },
            [&]() { XBase(target); }, true);
        return;
    }

    if (!freezeBasis) {
        TransformBasis1Qb(false, control);
        // TODO: Only controlled-by, for control?
        RevertBasis2Qb(control, true, false, { target });
        RevertBasis2Qb(target, true, true);
        tShard.AddInversionAngles(&cShard, 0, 0);
        return;
    }

    CTRLED_PHASE_INVERT_WRAP(
        CNOT(CTRL_1_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS), X(target), false, true, ONE_R1, ONE_R1);
}

void QUnit::AntiCNOT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(
        AntiCNOT(CTRL_1_ARGS), ApplyAntiControlledSingleBit(CTRL_GEN_ARGS), X(target), true, true, ONE_R1, ONE_R1);
}

void QUnit::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };

    if (TryCnotOptimize(controls, 2, target, ONE_CMPLX, ONE_CMPLX, false)) {
        return;
    }

    ApplyEitherControlled(controls, 2, { target }, false,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPlusMinus) {
                if (mappedControls.size() == 2) {
                    unit->ApplyControlledSinglePhase(
                        &(mappedControls[0]), mappedControls.size(), shards[target].mapped, ONE_CMPLX, -ONE_CMPLX);
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
    bitLenInt controls[2] = { control1, control2 };

    if (TryCnotOptimize(controls, 2, target, ONE_CMPLX, ONE_CMPLX, true)) {
        return;
    }

    ApplyEitherControlled(controls, 2, { target }, true,
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) {
            if (shards[target].isPlusMinus) {
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
    QEngineShard& tShard = shards[target];
    QEngineShard& cShard = shards[control];

    if (CACHED_ZERO(target) || CACHED_ZERO(control)) {
        return;
    }

    if (cShard.isPlusMinus && !tShard.isPlusMinus) {
        std::swap(control, target);
    }

    if (!freezeBasis) {
        TransformBasis1Qb(false, control);
        tShard.AddPhaseAngles(&cShard, 0, (real1)M_PI);
        return;
    }

    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;

    CTRLED_PHASE_INVERT_WRAP(
        CZ(CTRL_1_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS), Z(target), false, false, ONE_R1, -ONE_R1);
}

void QUnit::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (!PHASE_MATTERS(target)) {
        return;
    }

    RevertBasis2Qb(target, true);

    // Not necessary to check return after the above revert:
    shard.TryCommutePhase(topLeft, bottomRight);

    if (!shard.isPlusMinus) {
        // If the target bit is in a |0>/|1> eigenstate, this gate has no effect.
        if (PHASE_MATTERS(target)) {
            ApplyOrEmulate(
                shard, [&](QEngineShard& shard) { shard.unit->ApplySinglePhase(topLeft, bottomRight, shard.mapped); });
            shard.amp0 *= topLeft;
            shard.amp1 *= bottomRight;
            shard.ClampAmps();
        }
    } else {
        complex mtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
        TransformPhase(topLeft, bottomRight, mtrx);

        ApplyOrEmulate(shard, [&](QEngineShard& shard) { shard.unit->ApplySingleBit(mtrx, shard.mapped); });

        complex Y0 = shard.amp0;

        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        shard.ClampAmps();
    }

    CheckShardSeparable(target);
}

void QUnit::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
{
    QEngineShard& shard = shards[target];

    if (!PHASE_MATTERS(target) || (randGlobalPhase && (topRight == bottomLeft))) {
        X(target);
        return;
    }

    RevertBasis2Qb(target, false, true);

    // Not necessary to check return after the above revert:
    shard.TryFlipPhaseAnti();

    if (!shard.isPlusMinus) {
        ApplyOrEmulate(
            shard, [&](QEngineShard& shard) { shard.unit->ApplySingleInvert(topRight, bottomLeft, shard.mapped); });

        complex tempAmp1 = shard.amp0 * bottomLeft;
        shard.amp0 = shard.amp1 * topRight;
        shard.amp1 = tempAmp1;
        shard.ClampAmps();
    } else {
        complex mtrx[4] = { ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ZERO_CMPLX };
        TransformInvert(topRight, bottomLeft, mtrx);

        ApplyOrEmulate(shard, [&](QEngineShard& shard) { shard.unit->ApplySingleBit(mtrx, shard.mapped); });

        complex Y0 = shard.amp0;

        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        shard.ClampAmps();
    }

    CheckShardSeparable(target);
}

void QUnit::ApplyControlledSinglePhase(const bitLenInt* cControls, const bitLenInt& controlLen,
    const bitLenInt& cTarget, const complex topLeft, const complex bottomRight)
{
    // Commutes with controlled phase optimizations
    if (controlLen == 0) {
        ApplySinglePhase(topLeft, bottomRight, cTarget);
        return;
    }

    bitLenInt* controls = new bitLenInt[controlLen];
    std::copy(cControls, cControls + controlLen, controls);
    bitLenInt target = cTarget;

    if (IS_ONE_CMPLX(bottomRight) && CACHED_ONE(target)) {
        delete[] controls;
        return;
    }

    if (IS_ONE_CMPLX(topLeft)) {
        if (CACHED_ZERO(target)) {
            delete[] controls;
            return;
        }

        if (controlLen == 1U) {
            if (IS_ONE_CMPLX(-bottomRight)) {
                CZ(controls[0], target);
                delete[] controls;
                return;
            }
        }

        if (!shards[target].isPlusMinus) {
            for (bitLenInt i = 0; i < controlLen; i++) {
                if (shards[controls[i]].isPlusMinus) {
                    std::swap(controls[i], target);
                    break;
                }
            }
        }
    }

    QEngineShard& tShard = shards[target];
    QEngineShard& cShard = shards[controls[0]];

    if (!freezeBasis && (controlLen == 1U)) {
        TransformBasis1Qb(false, controls[0]);
        tShard.AddPhaseAngles(&cShard, (real1)arg(topLeft), (real1)arg(bottomRight));
        delete[] controls;
        return;
    }

    CTRLED_PHASE_INVERT_WRAP(ApplyControlledSinglePhase(CTRL_P_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS),
        ApplySinglePhase(topLeft, bottomRight, target), false, false, topLeft, bottomRight);

    delete[] controls;
}

void QUnit::ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex topRight, const complex bottomLeft)
{
    if (TryCnotOptimize(controls, controlLen, target, bottomLeft, topRight, false)) {
        return;
    }

    if (!freezeBasis && (controlLen == 1U)) {
        TransformBasis1Qb(false, controls[0]);
        // TODO: Only controlled-by, for control?
        RevertBasis2Qb(controls[0], true, false, { target });
        RevertBasis2Qb(target, true, true);

        shards[target].AddInversionAngles(&(shards[controls[0]]), (real1)arg(topRight), (real1)arg(bottomLeft));
        return;
    }

    CTRLED_PHASE_INVERT_WRAP(ApplyControlledSingleInvert(CTRL_I_ARGS), ApplyControlledSingleBit(CTRL_GEN_ARGS),
        ApplySingleInvert(topRight, bottomLeft, target), false, true, topRight, bottomLeft);
}

void QUnit::ApplyAntiControlledSinglePhase(const bitLenInt* cControls, const bitLenInt& controlLen,
    const bitLenInt& cTarget, const complex topLeft, const complex bottomRight)
{
    // Commutes with controlled phase optimizations
    if (controlLen == 0) {
        ApplySinglePhase(topLeft, bottomRight, cTarget);
        return;
    }

    bitLenInt* controls = new bitLenInt[controlLen];
    std::copy(cControls, cControls + controlLen, controls);
    bitLenInt target = cTarget;

    if ((IS_ONE_CMPLX(topLeft) && CACHED_ZERO(cTarget)) || (IS_ONE_CMPLX(bottomRight) && CACHED_ONE(cTarget))) {
        delete[] controls;
        return;
    }

    CTRLED_PHASE_INVERT_WRAP(ApplyAntiControlledSinglePhase(CTRL_P_ARGS), ApplyAntiControlledSingleBit(CTRL_GEN_ARGS),
        ApplySinglePhase(topLeft, bottomRight, target), true, false, topLeft, bottomRight);

    delete[] controls;
}

void QUnit::ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    if (TryCnotOptimize(controls, controlLen, target, bottomLeft, topRight, true)) {
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

    if ((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) {
        ApplySinglePhase(mtrx[0], mtrx[3], target);
        return;
    }
    if ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)) {
        ApplySingleInvert(mtrx[1], mtrx[2], target);
        return;
    }
    if ((randGlobalPhase || (mtrx[0] == complex(M_SQRT1_2, ZERO_R1))) && (mtrx[0] == mtrx[1]) && (mtrx[0] == mtrx[2]) &&
        (mtrx[2] == -mtrx[3])) {
        H(target);
        return;
    }

    QEngineShard& shard = shards[target];

    RevertBasis2Qb(target);

    complex trnsMtrx[4];

    if (!shard.isPlusMinus) {
        std::copy(mtrx, mtrx + 4, trnsMtrx);
    } else {
        Transform2x2(mtrx, trnsMtrx);
    }

    ApplyOrEmulate(shard, [&](QEngineShard& shard) { shard.unit->ApplySingleBit(trnsMtrx, shard.mapped); });

    if (DIRTY(shard)) {
        shard.isProbDirty = true;
        shard.isPhaseDirty = true;
        return;
    }

    complex Y0 = shard.amp0;

    shard.amp0 = (trnsMtrx[0] * Y0) + (trnsMtrx[1] * shard.amp1);
    shard.amp1 = (trnsMtrx[2] * Y0) + (trnsMtrx[3] * shard.amp1);
    shard.ClampAmps();

    CheckShardSeparable(target);
}

void QUnit::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (IsIdentity(mtrx, true)) {
        return;
    }

    if ((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) {
        ApplyControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
        return;
    }

    if ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)) {
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

    if ((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) {
        ApplyAntiControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
        return;
    }

    if ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)) {
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

#define CHECK_BREAK_AND_TRIM()                                                                                         \
    /* Check whether the bit probability is 0, (or 1, if "anti"). */                                                   \
    bitProb = inCurrentBasis ? ProbBase(controls[i]) : Prob(controls[i]);                                              \
    if (bitProb < min_norm) {                                                                                          \
        if (!anti) {                                                                                                   \
            /* This gate does nothing, so return without applying anything. */                                         \
            return;                                                                                                    \
        }                                                                                                              \
        /* This control has 100% chance to "fire," so don't entangle it. */                                            \
    } else if ((ONE_R1 - bitProb) < min_norm) {                                                                        \
        if (anti) {                                                                                                    \
            /* This gate does nothing, so return without applying anything. */                                         \
            return;                                                                                                    \
        }                                                                                                              \
        /* This control has 100% chance to "fire," so don't entangle it. */                                            \
    } else {                                                                                                           \
        if (!inCurrentBasis) {                                                                                         \
            ToPermBasis(controls[i]);                                                                                  \
        }                                                                                                              \
        controlVec.push_back(controls[i]);                                                                             \
    }

template <typename CF, typename F>
void QUnit::ApplyEitherControlled(const bitLenInt* controls, const bitLenInt& controlLen,
    const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F fn, const bool& inCurrentBasis)
{
    bitLenInt i, j;

    // If the controls start entirely separated from the targets, it's probably worth checking to see if the have
    // total or no probability of altering the targets, such that we can still keep them separate.

    std::vector<bitLenInt> controlVec;

    bool isSeparated = true;
    real1 bitProb;
    for (i = 0; i < controlLen; i++) {
        // If the shard's probability is cached, then it's free to check it, so we advance the loop.
        if (!shards[controls[i]].isProbDirty) {
            // This might determine that we can just skip out of the whole gate, in which case it returns this
            // method:
            CHECK_BREAK_AND_TRIM();
        } else {
            isSeparated = true;
            for (j = 0; j < targets.size(); j++) {
                // If the shard doesn't have a cached probability, and if it's in the same shard unit as any of the
                // targets, it isn't worth trying the next optimization.
                if (shards[controls[i]].unit == shards[targets[j]].unit) {
                    isSeparated = false;
                    break;
                }
            }
            if (isSeparated) {
                CHECK_BREAK_AND_TRIM();
            } else {
                if (!inCurrentBasis) {
                    ToPermBasis(controls[i]);
                }
                controlVec.push_back(controls[i]);
            }
        }
    }

    for (i = 0; i < targets.size(); i++) {
        RevertBasis2Qb(targets[i]);
    }

    if (controlVec.size() == 0) {
        // Here, the gate is guaranteed to act as if it wasn't controlled, so we apply the gate without controls,
        // avoiding an entangled representation.
        fn();

        return;
    }

    // TODO: If controls that survive the "first order" check above start out entangled,
    // then it might be worth checking whether there is any intra-unit chance of control bits
    // being conditionally all 0 or all 1, in any unit, due to entanglement.

    // If we've made it this far, we have to form the entangled representation and apply the gate.
    std::vector<bitLenInt> allBits(controlVec.size() + targets.size());
    std::copy(controlVec.begin(), controlVec.end(), allBits.begin());
    std::copy(targets.begin(), targets.end(), allBits.begin() + controlVec.size());
    // (Incidentally, we sort for the efficiency of QUnit's limited "mapper," a 1 dimensional array of qubits without
    // nearest neighbor restriction.)
    std::sort(allBits.begin(), allBits.end());

    std::vector<bitLenInt*> ebits(allBits.size());
    for (i = 0; i < ebits.size(); i++) {
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
    // target bit in |+>/|-> basis and acting as if |0>/|1> basis by commutation).
    cfn(unit, controlsMapped);

    for (i = 0; i < targets.size(); i++) {
        shards[targets[i]].isProbDirty = true;
        shards[targets[i]].isPhaseDirty = true;
    }
}

template <typename F>
void QUnit::CBoolReg(const bitLenInt& qInputStart, const bitCapInt& classicalInput, const bitLenInt& outputStart,
    const bitLenInt& length, F fn)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = bitSlice(i, classicalInput) != 0;
        fn(qInputStart + i, cBit, outputStart + i);
    }
}

void QUnit::AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            AND(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

void QUnit::OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            OR(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

void QUnit::XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    for (bitLenInt i = 0; i < length; i++) {
        XOR(inputStart1 + i, inputStart2 + i, outputStart + i);
    }
}

void QUnit::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    CBoolReg(qInputStart, classicalInput, outputStart, length,
        [&](bitLenInt qb, bool cb, bitLenInt l) { CLAND(qb, cb, l); });
}

void QUnit::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    CBoolReg(
        qInputStart, classicalInput, outputStart, length, [&](bitLenInt qb, bool cb, bitLenInt l) { CLOR(qb, cb, l); });
}

void QUnit::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    CBoolReg(qInputStart, classicalInput, outputStart, length,
        [&](bitLenInt qb, bool cb, bitLenInt l) { CLXOR(qb, cb, l); });
}

bool QUnit::CArithmeticOptimize(bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>* controlVec)
{
    if (controlLen == 0) {
        return false;
    }

    for (bitLenInt i = 0; i < controlLen; i++) {
        // If any control has a cached zero probability, this gate will do nothing, and we can avoid basically all
        // overhead.
        if (!shards[controls[i]].isProbDirty && (Prob(controls[i]) < min_norm)) {
            return true;
        }
    }

    controlVec->resize(controlLen);
    std::copy(controls, controls + controlLen, controlVec->begin());
    bitLenInt controlIndex = 0;

    for (bitLenInt i = 0; i < controlLen; i++) {
        real1 prob = Prob(controls[i]);
        if (prob < min_norm) {
            // If any control has zero probability, this gate will do nothing.
            return true;
        } else if ((ONE_R1 - prob) < min_norm) {
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

    INT(toMod, start, length, 0xFF, false, lControls, controlVec.size());

    delete[] lControls;
}

void QUnit::INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    EntangleRange(start, length);

    std::vector<bitLenInt> bits = { start, flagIndex };
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits = { &bits[0], &bits[1] };

    QInterfacePtr unit = Entangle(ebits);

    ((*unit).*fn)(toMod, shards[start].mapped, length, shards[flagIndex].mapped);

    DirtyShardRange(start, length);
    shards[flagIndex].isProbDirty = true;
    shards[flagIndex].isPhaseDirty = true;
}

void QUnit::INCxx(
    INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index)
{
    /* Make sure the flag bits are entangled in the same QU. */
    EntangleRange(start, length);
    std::vector<bitLenInt> bits = { start, flag1Index, flag2Index };
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits = { &bits[0], &bits[1], &bits[2] };

    QInterfacePtr unit = Entangle(ebits);

    ((*unit).*fn)(toMod, shards[start].mapped, length, shards[flag1Index].mapped, shards[flag2Index].mapped);

    DirtyShardRange(start, length);
    shards[flag1Index].isProbDirty = true;
    shards[flag2Index].isProbDirty = true;
    shards[flag1Index].isPhaseDirty = true;
    shards[flag2Index].isPhaseDirty = true;
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
    if (toMod == 0) {
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
                EntangleRange(start, partLength);
                DirtyShardRange(start, partLength);
                if (controlLen) {
                    allBits[controlLen] = start;
                    ebits[controlLen] = &allBits[controlLen];
                    QInterfacePtr unit = Entangle(ebits);
                    DirtyShardIndexVector(allBits);
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

    if ((toMod == 0) && (length == 0)) {
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
            EntangleRange(start, length, carryIndex, 1);
            shards[start].unit->INCC(toMod, shards[start].mapped, length, shards[carryIndex].mapped);
            DirtyShardRange(start, length);
            DirtyShardRange(carryIndex, 1U);
        }
    } else {
        EntangleRange(start, length);
        DirtyShardRange(start, length);
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
    if (toMod == 0) {
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

void QUnit::INCBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    // BCD variants are low priority for optimization, for the time being.
    EntangleRange(start, length);
    shards[start].unit->INCBCD(toMod, shards[start].mapped, length);
    DirtyShardRange(start, length);
}

void QUnit::INCBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // BCD variants are low priority for optimization, for the time being.
    INCx(&QInterface::INCBCDC, toMod, start, length, carryIndex);
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

void QUnit::DECBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    // BCD variants are low priority for optimization, for the time being.
    EntangleRange(start, length);
    shards[start].unit->DECBCD(toMod, shards[start].mapped, length);
    DirtyShardRange(start, length);
}

void QUnit::DECBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // BCD variants are low priority for optimization, for the time being.
    INCx(&QInterface::DECBCDC, toMod, start, length, carryIndex);
}

void QUnit::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (toMul == 0U) {
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

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->MUL(toMul, shards[inOutStart].mapped, shards[carryStart].mapped, length);
    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);
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

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->DIV(toDiv, shards[inOutStart].mapped, shards[carryStart].mapped, length);
    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);
}

void QUnit::xMULModNOut(
    bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length, bool inverse)
{
    // Inexpensive edge case
    if (toMod == 0U) {
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

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inStart, length, outStart, length);
    if (inverse) {
        shards[inStart].unit->IMULModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    } else {
        shards[inStart].unit->MULModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    }
    DirtyShardRangePhase(inStart, length);
    DirtyShardRange(outStart, length);
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
    EntangleRange(start, length);
    EntangleRange(carryStart, length);
    DirtyShardRange(carryStart, length);

    std::vector<bitLenInt> bits(controlVec.size() + 2);
    for (bitLenInt i = 0; i < controlVec.size(); i++) {
        bits[i] = controlVec[i];
    }
    bits[controlVec.size()] = start;
    bits[controlVec.size() + 1] = carryStart;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(controlVec.size() + 2);
    for (bitLenInt i = 0; i < (controlVec.size() + 2); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = Entangle(ebits);

    controlsMapped->resize(controlVec.size() == 0 ? 1 : controlVec.size());
    for (bitLenInt i = 0; i < controlVec.size(); i++) {
        (*controlsMapped)[i] = shards[controlVec[i]].mapped;
        shards[controlVec[i]].isPhaseDirty = true;
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

    ((*unit).*fn)(
        toMod, shards[start].mapped, shards[carryStart].mapped, length, &(controlsMapped[0]), controlVec.size());

    DirtyShardRange(start, length);
}

void QUnit::CMULModx(CMULModFn fn, bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
    bitLenInt length, std::vector<bitLenInt> controlVec)
{
    std::vector<bitLenInt> controlsMapped;
    QInterfacePtr unit = CMULEntangle(controlVec, start, carryStart, length, &controlsMapped);

    ((*unit).*fn)(
        toMod, modN, shards[start].mapped, shards[carryStart].mapped, length, &(controlsMapped[0]), controlVec.size());

    DirtyShardRangePhase(start, length);
}

void QUnit::CMUL(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0U) {
        MUL(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QInterface::CMUL, toMod, start, carryStart, length, controls, controlLen);
}

void QUnit::CDIV(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0U) {
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

    if (controlVec.size() == 0U) {
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
    if (controlLen == 0U) {
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

void QUnit::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    bitLenInt min1 = length - 1U;
    bitLenInt* controls = new bitLenInt[min1];
    for (bitLenInt i = 0; i < min1; i++) {
        controls[i] = start + i + 1U;
    }
    ApplyAntiControlledSinglePhase(controls, min1, start, -ONE_CMPLX, ONE_CMPLX);
    delete[] controls;
}

void QUnit::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(start, length)) {
        if (GetCachedPermutation(start, length) < greaterPerm) {
            // This has no physical effect, but we do it to respect direct simulator check of amplitudes:
            ApplyOrEmulate(shards[start], [&](QEngineShard& shard) { shard.unit->PhaseFlip(); });
        }
        return;
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(start, length);
    shards[start].unit->PhaseFlipIfLess(greaterPerm, shards[start].mapped, length);
    DirtyShardRange(start, length);
}

void QUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (!shards[flagIndex].isProbDirty) {
        real1 prob = Prob(flagIndex);
        if (prob < min_norm) {
            return;
        } else if ((ONE_R1 - prob) < min_norm) {
            PhaseFlipIfLess(greaterPerm, start, length);
            return;
        }
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(start, length);

    std::vector<bitLenInt> bits = { start, flagIndex };
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits = { &bits[0], &bits[1] };

    QInterfacePtr unit = Entangle(ebits);

    unit->CPhaseFlipIfLess(greaterPerm, shards[start].mapped, length, shards[flagIndex].mapped);

    DirtyShardRange(start, length);
    shards[flagIndex].isPhaseDirty = true;
}

void QUnit::PhaseFlip()
{
    QEngineShard& shard = shards[0];
    if (PHASE_MATTERS(0)) {
        TransformBasis1Qb(false, 0);
        ApplyOrEmulate(shard, [&](QEngineShard& shard) { shard.unit->PhaseFlip(); });
        shard.amp1 = -shard.amp1;
    }
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
    shards[carryIndex].isProbDirty = true;
    shards[carryIndex].isPhaseDirty = true;

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
    shards[carryIndex].isProbDirty = true;
    shards[carryIndex].isPhaseDirty = true;

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

    EntangleRange(start, length);

    shards[start].unit->Hash(shards[start].mapped, length, values);

    DirtyShardRangePhase(start, length);
}

bool QUnit::ParallelUnitApply(ParallelUnitFn fn, real1 param1, real1 param2)
{
    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (find(units.begin(), units.end(), toFind) == units.end()) {
            units.push_back(toFind);
            if (!fn(toFind, param1, param2)) {
                return false;
            }
        }
    }

    return true;
}

void QUnit::UpdateRunningNorm(real1 norm_thresh)
{
    EndAllEmulation();
    ParallelUnitApply(
        [](QInterfacePtr unit, real1 norm_thresh, real1 unused2) {
            unit->UpdateRunningNorm(norm_thresh);
            return true;
        },
        norm_thresh);
}

void QUnit::NormalizeState(real1 nrm, real1 norm_thresh)
{
    EndAllEmulation();
    ParallelUnitApply(
        [](QInterfacePtr unit, real1 nrm, real1 norm_thresh) {
            unit->NormalizeState(nrm, norm_thresh);
            return true;
        },
        nrm, norm_thresh);
}

void QUnit::Finish()
{
    ParallelUnitApply([](QInterfacePtr unit, real1 unused1, real1 unused2) {
        unit->Finish();
        return true;
    });
}

bool QUnit::isFinished()
{
    return ParallelUnitApply([](QInterfacePtr unit, real1 unused1, real1 unused2) { return unit->isFinished(); });
}

bool QUnit::ApproxCompare(QUnitPtr toCompare)
{
    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        return false;
    }

    EndAllEmulation();

    QUnitPtr thisCopy = std::dynamic_pointer_cast<QUnit>(Clone());
    thisCopy->EntangleAll();
    thisCopy->OrderContiguous(thisCopy->shards[0].unit);

    QUnitPtr thatCopy = std::dynamic_pointer_cast<QUnit>(toCompare->Clone());
    thatCopy->EntangleAll();
    thatCopy->OrderContiguous(thatCopy->shards[0].unit);

    return thisCopy->shards[0].unit->ApproxCompare(thatCopy->shards[0].unit);
}

QInterfacePtr QUnit::Clone()
{
    // TODO: Copy buffers instead of flushing?
    ToPermBasisAll();
    EndAllEmulation();

    QUnitPtr copyPtr = std::make_shared<QUnit>(engine, subengine, qubitCount, 0, rand_generator,
        complex(ONE_R1, ZERO_R1), doNormalize, randGlobalPhase, useHostRam);

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

void QUnit::TransformBasis1Qb(const bool& toPlusMinus, const bitLenInt& i)
{
    if (freezeBasis || (toPlusMinus == shards[i].isPlusMinus)) {
        // Recursive call that should be blocked,
        // or already in target basis.
        return;
    }

    freezeBasis = true;
    H(i);
    shards[i].isPlusMinus = toPlusMinus;
    freezeBasis = false;
}

void QUnit::RevertBasis2Qb(const bitLenInt& i, const bool& onlyInvert, const bool& onlyControlling,
    std::set<bitLenInt> exceptControlling, std::set<bitLenInt> exceptTargetedBy, const bool& dumpSkipped)
{
    QEngineShard& shard = shards[i];

    if (freezeBasis || !QUEUED_PHASE(shard)) {
        // Recursive call that should be blocked,
        // or already in target basis.
        return;
    }

    shard.CombineGates();

    if (onlyControlling && !onlyInvert) {
        shard.OptimizeControls();
    }

    bitLenInt controls[1];
    complex polar0, polar1;
    complex mtrx[4];

    ShardToPhaseMap::iterator phaseShard;
    controls[0] = i;
    ShardToPhaseMap controlsShards = shard.controlsShards;
    while (controlsShards.size() > 0) {
        phaseShard = controlsShards.begin();
        QEngineShardPtr partner = phaseShard->first;

        if (onlyInvert && !phaseShard->second->isInvert) {
            controlsShards.erase(phaseShard);
            if (dumpSkipped) {
                shard.RemovePhaseTarget(partner);
            }
            continue;
        }

        bitLenInt j = FindShardIndex(*partner);

        if (exceptControlling.find(j) != exceptControlling.end()) {
            controlsShards.erase(phaseShard);
            if (dumpSkipped) {
                shard.RemovePhaseTarget(partner);
            }
            continue;
        }

        polar0 = std::polar(ONE_R1, phaseShard->second->angle0);
        polar1 = std::polar(ONE_R1, phaseShard->second->angle1);

        if (shards[j].isPlusMinus) {
            if (phaseShard->second->isInvert) {
                mtrx[0] = (polar0 + polar1) / (ONE_R1 * 2);
                mtrx[1] = (-polar0 + polar1) / (ONE_R1 * 2);
                mtrx[2] = (polar0 - polar1) / (ONE_R1 * 2);
                mtrx[3] = (-polar0 - polar1) / (ONE_R1 * 2);
            } else {
                mtrx[0] = (polar0 + polar1) / (ONE_R1 * 2);
                mtrx[1] = (polar0 - polar1) / (ONE_R1 * 2);
                mtrx[2] = mtrx[1];
                mtrx[3] = mtrx[0];
            }

            shards[j].isPlusMinus = false;
            freezeBasis = true;
            ApplyControlledSingleBit(controls, 1U, j, mtrx);
            freezeBasis = false;
            shards[j].isPlusMinus = true;
        } else {
            freezeBasis = true;
            if (phaseShard->second->isInvert) {
                ApplyControlledSingleInvert(controls, 1U, j, polar0, polar1);
            } else {
                ApplyControlledSinglePhase(controls, 1U, j, polar0, polar1);
            }
            freezeBasis = false;
        }

        controlsShards.erase(phaseShard);
        shard.RemovePhaseTarget(partner);
    }

    if (onlyControlling) {
        return;
    }

    ShardToPhaseMap targetOfShards = shard.targetOfShards;
    while (targetOfShards.size() > 0) {
        phaseShard = targetOfShards.begin();
        QEngineShardPtr partner = phaseShard->first;

        if (onlyInvert && !phaseShard->second->isInvert) {
            targetOfShards.erase(phaseShard);
            if (dumpSkipped) {
                shard.RemovePhaseControl(partner);
            }
            continue;
        }

        bitLenInt j = FindShardIndex(*partner);

        if (exceptTargetedBy.find(j) != exceptTargetedBy.end()) {
            targetOfShards.erase(phaseShard);
            if (dumpSkipped) {
                shard.RemovePhaseControl(partner);
            }
            continue;
        }

        controls[0] = j;

        polar0 = std::polar(ONE_R1, phaseShard->second->angle0);
        polar1 = std::polar(ONE_R1, phaseShard->second->angle1);

        if (shards[i].isPlusMinus) {
            if (phaseShard->second->isInvert) {
                mtrx[0] = (polar0 + polar1) / (ONE_R1 * 2);
                mtrx[1] = (-polar0 + polar1) / (ONE_R1 * 2);
                mtrx[2] = (polar0 - polar1) / (ONE_R1 * 2);
                mtrx[3] = (-polar0 - polar1) / (ONE_R1 * 2);
            } else {
                mtrx[0] = (polar0 + polar1) / (ONE_R1 * 2);
                mtrx[1] = (polar0 - polar1) / (ONE_R1 * 2);
                mtrx[2] = mtrx[1];
                mtrx[3] = mtrx[0];
            }

            shards[i].isPlusMinus = false;
            freezeBasis = true;
            ApplyControlledSingleBit(controls, 1U, i, mtrx);
            freezeBasis = false;
            shards[i].isPlusMinus = true;
        } else {
            freezeBasis = true;
            if (phaseShard->second->isInvert) {
                ApplyControlledSingleInvert(controls, 1U, i, polar0, polar1);
            } else {
                ApplyControlledSinglePhase(controls, 1U, i, polar0, polar1);
            }
            freezeBasis = false;
        }

        targetOfShards.erase(phaseShard);
        shard.RemovePhaseControl(partner);
    }
}

void QUnit::CheckShardSeparable(const bitLenInt& target)
{
    QEngineShard& shard = shards[target];

    if (shard.isProbDirty || (shard.unit->GetQubitCount() == 1U)) {
        return;
    }

    if (norm(shard.amp0) < min_norm) {
        SeparateBit(true, target);
    } else if (norm(shard.amp1) < min_norm) {
        SeparateBit(false, target);
    }
}

} // namespace Qrack
