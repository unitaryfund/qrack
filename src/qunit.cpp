//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <ctime>
#include <initializer_list>
#include <map>

#include "qfactory.hpp"
#include "qunit.hpp"

namespace Qrack {

QUnit::QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID, bool useHardwareRNG)
    : QUnit(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceID,
          useHardwareRNG)
{
    // Intentionally left blank
}

QUnit::QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceID,
    bool useHardwareRNG)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG)
    , engine(eng)
    , subengine(subEng)
    , devID(deviceID)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , randGlobalPhase(randomGlobalPhase)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
{
    shards.resize(qBitCount);

    SetPermutation(initState, phaseFactor);
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool bitState;

    Finish();

    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((1 << i) & perm) >> i;
        shards[i].unit = CreateQuantumInterface(engine, subengine, 1U, bitState ? 1U : 0U, rand_generator, phaseFac,
            doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND);
        shards[i].mapped = 0;
        shards[i].prob = bitState ? ONE_R1 : ZERO_R1;
        shards[i].isProbDirty = false;
        shards[i].phase = ZERO_R1;
        shards[i].isPhaseDirty = false;
    }
}

void QUnit::CopyState(QUnitPtr orig) { CopyState(orig.get()); }

// protected method
void QUnit::CopyState(QUnit* orig)
{
    SetQubitCount(orig->GetQubitCount());
    shards.clear();

    /* Set up the shards to refer to the new unit. */
    std::map<QInterfacePtr, QInterfacePtr> otherUnits;
    for (auto&& otherShard : orig->shards) {
        QEngineShard shard;
        shard.mapped = otherShard.mapped;
        shard.prob = otherShard.prob;
        shard.isProbDirty = otherShard.isProbDirty;
        shard.phase = otherShard.phase;
        shard.isPhaseDirty = otherShard.isPhaseDirty;
        if (otherUnits.find(otherShard.unit) == otherUnits.end()) {
            otherUnits[otherShard.unit] = CreateQuantumInterface(engine, subengine, 1, 0, rand_generator, phaseFactor,
                doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND);
            otherUnits[otherShard.unit]->CopyState(otherShard.unit);
        }
        shard.unit = otherUnits[otherShard.unit];
        shards.push_back(shard);
    }
}

void QUnit::CopyState(QInterfacePtr orig)
{
    QInterfacePtr unit = CreateQuantumInterface(engine, subengine, orig->GetQubitCount(), 0, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND);
    unit->CopyState(orig);

    SetQubitCount(orig->GetQubitCount());
    shards.clear();

    /* Set up the shards to refer to the new unit. */
    for (bitLenInt i = 0; i < (orig->GetQubitCount()); i++) {
        QEngineShard shard;
        shard.unit = unit;
        shard.mapped = i;
        shard.isProbDirty = true;
        shard.isPhaseDirty = true;
        shards.push_back(shard);
    }
}

void QUnit::SetQuantumState(const complex* inputState)
{
    auto unit = CreateQuantumInterface(engine, subengine, qubitCount, 0, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND);
    unit->SetQuantumState(inputState);

    int idx = 0;
    for (auto&& shard : shards) {
        shard.unit = unit;
        shard.mapped = idx++;
        shard.isProbDirty = true;
        shard.isPhaseDirty = true;
    }
}

void QUnit::GetQuantumState(complex* outputState)
{
    QUnit qUnitCopy(engine, subengine, 1, 0);
    qUnitCopy.CopyState((QUnit*)this);
    qUnitCopy.OrderContiguous(qUnitCopy.EntangleAll());
    qUnitCopy.shards[0].unit->GetQuantumState(outputState);
}

void QUnit::GetProbs(real1* outputProbs)
{
    QUnit qUnitCopy(engine, subengine, 1, 0);
    qUnitCopy.CopyState((QUnit*)this);
    qUnitCopy.OrderContiguous(qUnitCopy.EntangleAll());
    qUnitCopy.shards[0].unit->GetProbs(outputProbs);
}

complex QUnit::GetAmplitude(bitCapInt perm)
{
    complex result(ONE_R1, ZERO_R1);

    std::map<QInterfacePtr, bitCapInt> perms;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (perms.find(shards[i].unit) == perms.end()) {
            perms[shards[i].unit] = 0U;
        }
        if (perm & (1U << i)) {
            perms[shards[i].unit] |= 1U << shards[i].mapped;
        }
    }

    for (auto&& qi : perms) {
        result *= qi.first->GetAmplitude(qi.second);
    }

    if (shards[0].unit->GetQubitCount() > 1) {
        if (norm(result) > (ONE_R1 - min_norm)) {
            SetPermutation(perm);
        }
    }

    return result;
}

bitLenInt QUnit::Compose(QUnitPtr toCopy) { return Compose(toCopy, qubitCount); }

/*
 * Append QInterface in the middle of QUnit.
 */
bitLenInt QUnit::Compose(QUnitPtr toCopy, bitLenInt start)
{
    /* Create a clone of the quantum state in toCopy. */
    QUnitPtr clone(toCopy);

    /* Insert the new shards in the middle */
    shards.insert(shards.begin() + start, clone->shards.begin(), clone->shards.end());

    SetQubitCount(qubitCount + toCopy->GetQubitCount());

    return start;
}

void QUnit::Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
{
    /* TODO: This method should compose the bits for the destination without cohering the length first */

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
            dest->shards[start + i].prob = shards[start + i].prob;
            dest->shards[start + i].isProbDirty = shards[start + i].isProbDirty;
            dest->shards[start + i].phase = shards[start + i].phase;
            dest->shards[start + i].isPhaseDirty = shards[start + i].isPhaseDirty;
        }

        if (unit->GetQubitCount() > length) {
            unit->Decompose(mapped, length, destEngine);
        } else {
            destEngine->CopyState(unit);
        }
    } else {
        unit->Dispose(mapped, length);
    }

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

QInterfacePtr QUnit::EntangleIterator(std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    std::vector<QInterfacePtr> units;
    units.reserve((int)(last - first));

    QInterfacePtr unit1 = shards[**first].unit;
    std::map<QInterfacePtr, bool> found;

    found[unit1] = true;

    /* Walk through all of the supplied bits and create a unique list to compose. */
    for (auto bit = first + 1; bit < last; bit++) {
        if (found.find(shards[**bit].unit) == found.end()) {
            found[shards[**bit].unit] = true;
            units.push_back(shards[**bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    if (units.size() != 0) {
        auto&& offsets = unit1->Compose(units);

        /* Since each unit will be collapsed in-order, one set of bits at a time. */
        for (auto&& shard : shards) {
            auto search = offsets.find(shard.unit);
            if (search != offsets.end()) {
                shard.mapped += search->second;
                shard.unit = unit1;
            }
        }
    }

    /* Change the source parameters to the correct newly mapped bit indexes. */
    for (auto bit = first; bit < last; bit++) {
        **bit = shards[**bit].mapped;
    }

    return unit1;
}

QInterfacePtr QUnit::Entangle(std::vector<bitLenInt*> bits) { return EntangleIterator(bits.begin(), bits.end()); }

QInterfacePtr QUnit::EntangleRange(bitLenInt start, bitLenInt length)
{
    if (length == 1) {
        return shards[start].unit;
    }

    std::vector<bitLenInt> bits(length);
    std::vector<bitLenInt*> ebits(length);
    for (auto i = 0; i < length; i++) {
        bits[i] = i + start;
        ebits[i] = &bits[i];
    }

    QInterfacePtr toRet = EntangleIterator(ebits.begin(), ebits.end());
    OrderContiguous(shards[start].unit);
    return toRet;
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2)
{
    std::vector<bitLenInt> bits(length1 + length2);
    std::vector<bitLenInt*> ebits(length1 + length2);

    for (auto i = 0; i < length1; i++) {
        bits[i] = i + start1;
        ebits[i] = &bits[i];
    }

    for (auto i = 0; i < length2; i++) {
        bits[i + length1] = i + start2;
        ebits[i + length1] = &bits[i + length1];
    }

    QInterfacePtr toRet = EntangleIterator(ebits.begin(), ebits.end());
    OrderContiguous(shards[start1].unit);
    return toRet;
}

QInterfacePtr QUnit::EntangleRange(
    bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3)
{
    std::vector<bitLenInt> bits(length1 + length2 + length3);
    std::vector<bitLenInt*> ebits(length1 + length2 + length3);

    for (auto i = 0; i < length1; i++) {
        bits[i] = i + start1;
        ebits[i] = &bits[i];
    }

    for (auto i = 0; i < length2; i++) {
        bits[i + length1] = i + start2;
        ebits[i + length1] = &bits[i + length1];
    }

    for (auto i = 0; i < length3; i++) {
        bits[i + length1 + length2] = i + start3;
        ebits[i + length1 + length2] = &bits[i + length1 + length2];
    }

    QInterfacePtr toRet = EntangleIterator(ebits.begin(), ebits.end());
    OrderContiguous(shards[start1].unit);
    return toRet;
}

QInterfacePtr QUnit::EntangleAll()
{
    std::vector<QInterfacePtr> units;
    units.reserve(qubitCount);

    QInterfacePtr unit1 = shards[0].unit;
    std::map<QInterfacePtr, bool> found;

    found[unit1] = true;

    /* Walk through all of the supplied bits and create a unique list to compose. */
    for (bitLenInt bit = 1; bit < qubitCount; bit++) {
        if (found.find(shards[bit].unit) == found.end()) {
            found[shards[bit].unit] = true;
            units.push_back(shards[bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    if (units.size() != 0) {
        auto&& offsets = unit1->QInterface::Compose(units);

        /* Since each unit will be collapsed in-order, one set of bits at a time. */
        for (auto&& shard : shards) {
            auto search = offsets.find(shard.unit);
            if (search != offsets.end()) {
                shard.mapped += search->second;
                shard.unit = unit1;
            }
        }
    }

    return unit1;
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
    }

    QInterfacePtr separatedBits = CreateQuantumInterface(engine, subengine, length, 0, rand_generator,
        complex(ONE_R1, ZERO_R1), doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND);

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

    int j = 0;
    for (int i = 0; i < qubitCount; i++) {
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
bool QUnit::CheckBitPermutation(bitLenInt qubitIndex)
{
    if (!shards[qubitIndex].isProbDirty &&
        ((Prob(qubitIndex) < min_norm) || ((ONE_R1 - Prob(qubitIndex)) < min_norm))) {
        return true;
    } else {
        return false;
    }
}

/// Check if all qubits in the range have cached probabilities indicating that they are in permutation basis
/// eigenstates, for optimization.
bool QUnit::CheckBitsPermutation(bitLenInt start, bitLenInt length)
{
    // Certain optimizations become obvious, if all bits in a range are in permutation basis eigenstates.
    // Then, operations can often be treated as classical, instead of quantum.
    for (bitLenInt i = 0; i < length; i++) {
        if (!CheckBitPermutation(start + i)) {
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
        if (shards[start + i].prob >= (ONE_R1 / 2)) {
            res |= 1U << i;
        }
    }
    return res;
}

void QUnit::DumpShards()
{
    int i = 0;
    for (auto shard : shards) {
        printf("%2d.\t%p[%d]\n", i++, shard.unit.get(), shard.mapped);
    }
}

real1 QUnit::Prob(bitLenInt qubit)
{
    QEngineShard& shard = shards[qubit];

    if (shard.isProbDirty) {
        shard.prob = (shard.unit->Prob)(shard.mapped);
        shard.isProbDirty = false;

        if (shard.unit->GetQubitCount() > 1) {
            if (shard.prob < min_norm) {
                SeparateBit(false, qubit);
            } else if (shard.prob > (ONE_R1 - min_norm)) {
                SeparateBit(true, qubit);
            }
        }
    }

    return shard.prob;
}

real1 QUnit::ProbAll(bitCapInt perm)
{
    real1 result = ONE_R1;

    std::map<QInterfacePtr, bitCapInt> perms;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (perms.find(shards[i].unit) == perms.end()) {
            perms[shards[i].unit] = 0U;
        }
        if (perm & (1U << i)) {
            perms[shards[i].unit] |= 1U << shards[i].mapped;
        }
    }

    for (auto&& qi : perms) {
        result *= qi.first->ProbAll(qi.second);
    }

    if (result > (ONE_R1 - min_norm)) {
        SetPermutation(perm);
        return ONE_R1;
    }

    return clampProb(result);
}

void QUnit::SeparateBit(bool value, bitLenInt qubit)
{
    QEngineShard origShard = shards[qubit];

    QInterfacePtr dest = CreateQuantumInterface(engine, subengine, 1, value ? 1 : 0, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND);

    origShard.unit->Dispose(origShard.mapped, 1);

    /* Update the mappings. */
    shards[qubit].unit = dest;
    shards[qubit].mapped = 0;
    shards[qubit].prob = value ? ONE_R1 : ZERO_R1;
    shards[qubit].isProbDirty = false;
    shards[qubit].phase = ZERO_R1;
    shards[qubit].isPhaseDirty = false;

    for (auto&& testShard : shards) {
        if (testShard.unit == origShard.unit && testShard.mapped > origShard.mapped) {
            testShard.mapped--;
        }
    }
}

bool QUnit::ForceM(bitLenInt qubit, bool res, bool doForce)
{
    bool result = shards[qubit].unit->ForceM(shards[qubit].mapped, res, doForce);

    if (shards[qubit].unit->GetQubitCount() == 1) {
        shards[qubit].prob = result ? ONE_R1 : ZERO_R1;
        shards[qubit].isProbDirty = false;
        shards[qubit].phase = ZERO_R1;
        shards[qubit].isPhaseDirty = false;

        /* If we're keeping the bits, and they're already in their own unit, there's nothing to do. */
        return result;
    }

    SeparateBit(result, qubit);

    return result;
}

/// Set register bits to given permutation
void QUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    MReg(start, length);

    bool bitState;
    for (bitLenInt i = 0; i < length; i++) {
        bitState = value & (1 << i);
        shards[i + start].unit->SetPermutation(bitState ? 1 : 0);
        shards[i + start].prob = bitState ? ONE_R1 : ZERO_R1;
        shards[i + start].isProbDirty = false;
        shards[i + start].phase = ZERO_R1;
        shards[i + start].isPhaseDirty = false;
    }
}

void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    // Swap the bit mapping.
    std::swap(shard1.mapped, shard2.mapped);

    // Swap the QInterface object.
    std::swap(shard1.unit, shard2.unit);

    // Swap the cached probability.
    std::swap(shard1.prob, shard2.prob);
    std::swap(shard1.isProbDirty, shard2.isProbDirty);
    std::swap(shard1.phase, shard2.phase);
    std::swap(shard1.isPhaseDirty, shard2.isPhaseDirty);
}

/* Unfortunately, many methods are overloaded, which prevents using just the address-to-member. */
#define PTR3(OP) (void (QInterface::*)(bitLenInt, bitLenInt, bitLenInt))(&QInterface::OP)
#define PTR2(OP) (void (QInterface::*)(bitLenInt, bitLenInt))(&QInterface::OP)
#define PTR1(OP) (void (QInterface::*)(bitLenInt))(&QInterface::OP)
#define PTR2A(OP) (void (QInterface::*)(real1, bitLenInt, bitLenInt))(&QInterface::OP)
#define PTRA(OP) (void (QInterface::*)(real1, bitLenInt))(&QInterface::OP)

void QUnit::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    shards[qubit1].isProbDirty = true;
    shards[qubit2].isProbDirty = true;
    shards[qubit1].isPhaseDirty = true;
    shards[qubit2].isPhaseDirty = true;

    EntangleAndCallMember(PTR2(SqrtSwap), qubit1, qubit2);
}

void QUnit::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    shards[qubit1].isProbDirty = true;
    shards[qubit2].isProbDirty = true;
    shards[qubit1].isPhaseDirty = true;
    shards[qubit2].isPhaseDirty = true;

    EntangleAndCallMember(PTR2(ISqrtSwap), qubit1, qubit2);
}

void QUnit::UniformlyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const complex* mtrxs)
{
    // TODO: Controls that have exactly 0 or 1 probability can be optimized out of the gate.

    // If there are no controls, this is equivalent to the single bit gate.
    if (controlLen == 0) {
        ApplySingleBit(mtrxs, true, qubitIndex);
        return;
    }

    bitLenInt i;

    std::vector<bitLenInt> bits(controlLen + 1);
    for (i = 0; i < controlLen; i++) {
        bits[i] = controls[i];
    }
    bits[controlLen] = qubitIndex;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(controlLen + 1);
    for (i = 0; i < bits.size(); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    bitLenInt* mappedControls = new bitLenInt[controlLen];
    for (i = 0; i < controlLen; i++) {
        mappedControls[i] = shards[controls[i]].mapped;
        shards[controls[i]].isPhaseDirty = true;
    }

    unit->UniformlyControlledSingleBit(mappedControls, controlLen, shards[qubitIndex].mapped, mtrxs);

    shards[qubitIndex].isProbDirty = true;
    shards[qubitIndex].isPhaseDirty = true;

    delete[] mappedControls;
}

void QUnit::H(bitLenInt target)
{
    QEngineShard& shard = shards[target];
    shard.unit->H(shard.mapped);

    if (shard.isProbDirty || shard.isPhaseDirty) {
        shard.isProbDirty = true;
        shard.isPhaseDirty = true;
        return;
    }

    real1 prob = shard.prob;
    real1 phase = shard.phase;

    complex zeroAmpIn = ((real1)sqrt(ONE_R1 - prob)) * complex(ONE_R1, ZERO_R1);
    complex oneAmpOut = ((real1)sqrt(prob)) * complex(cos(phase), sin(phase));

    complex zeroAmpOut = ((real1)M_SQRT1_2) * (zeroAmpIn + oneAmpOut);
    oneAmpOut = ((real1)M_SQRT1_2) * (zeroAmpIn - oneAmpOut);

    prob = norm(oneAmpOut);
    shard.prob = prob;
    shard.phase = ClampPhase(arg(oneAmpOut) - arg(zeroAmpOut));

    if (shard.unit->GetQubitCount() > 1) {
        if (prob < min_norm) {
            SeparateBit(false, target);
        } else if (prob > (ONE_R1 - min_norm)) {
            SeparateBit(true, target);
        }
    }
}

void QUnit::X(bitLenInt target)
{
    shards[target].unit->X(shards[target].mapped);
    shards[target].prob = ONE_R1 - shards[target].prob;
    shards[target].phase = ClampPhase(2 * M_PI - shards[target].phase);
}

void QUnit::Z(bitLenInt target)
{
    QEngineShard& shard = shards[target];
    // If the target bit is in a |0>/|1> eigenstate, this gate has no effect.
    if (shard.isProbDirty || !((shard.prob < min_norm) || ((ONE_R1 - shard.prob) < min_norm))) {
        shard.unit->Z(shard.mapped);
        shard.phase = ClampPhase(shard.phase + M_PI);
    }
}

#define CTRLED_CALL_WRAP(ctrld, bare, anti)                                                                            \
    ApplyEitherControlled(controls, controlLen, { target }, anti,                                                      \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->ctrld; }, [&]() { bare; })
#define CTRLED_SWAP_WRAP(ctrld, bare, anti)                                                                            \
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, anti,                                              \
        [&](QInterfacePtr unit, std::vector<bitLenInt> mappedControls) { unit->ctrld; }, [&]() { bare; })
#define CTRL_ARGS &(mappedControls[0]), controlLen, shards[target].mapped, mtrx
#define CTRL_1_ARGS mappedControls[0], shards[target].mapped
#define CTRL_2_ARGS mappedControls[0], mappedControls[1], shards[target].mapped
#define CTRL_S_ARGS &(mappedControls[0]), controlLen, shards[qubit1].mapped, shards[qubit2].mapped
#define CTRL_P_ARGS &(mappedControls[0]), controlLen, shards[target].mapped, topLeft, bottomRight
#define CTRL_I_ARGS &(mappedControls[0]), controlLen, shards[target].mapped, topRight, bottomLeft

void QUnit::CNOT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;
    CTRLED_CALL_WRAP(CNOT(CTRL_1_ARGS), X(target), false);
}

void QUnit::AntiCNOT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;
    CTRLED_CALL_WRAP(AntiCNOT(CTRL_1_ARGS), X(target), true);
}

void QUnit::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };
    bitLenInt controlLen = 2;
    CTRLED_CALL_WRAP(CCNOT(CTRL_2_ARGS), X(target), false);
}

void QUnit::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };
    bitLenInt controlLen = 2;
    CTRLED_CALL_WRAP(AntiCCNOT(CTRL_2_ARGS), X(target), true);
}

void QUnit::CZ(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    bitLenInt controlLen = 1;
    CTRLED_CALL_WRAP(CZ(CTRL_1_ARGS), Z(target), false);
}

void QUnit::ApplySinglePhase(const complex topLeft, const complex bottomRight, bool doCalcNorm, bitLenInt target)
{
    QEngineShard& shard = shards[target];
    // If the target bit is in a |0>/|1> eigenstate, this gate has no effect.
    if (shard.isProbDirty || !((shard.prob < min_norm) || ((ONE_R1 - shard.prob) < min_norm))) {
        shard.unit->ApplySinglePhase(topLeft, bottomRight, doCalcNorm, shard.mapped);
        shard.phase = ClampPhase(shard.phase + arg(bottomRight) - arg(topLeft));
    }
}

void QUnit::ApplySingleInvert(const complex topRight, const complex bottomLeft, bool doCalcNorm, bitLenInt target)
{
    shards[target].unit->ApplySingleInvert(topRight, bottomLeft, doCalcNorm, shards[target].mapped);
    shards[target].prob = ONE_R1 - shards[target].prob;
    shards[target].phase = ClampPhase((2 * M_PI - shards[target].phase) + (arg(topRight) - arg(bottomLeft)));
}

void QUnit::ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex topLeft, const complex bottomRight)
{
    QEngineShard& shard = shards[target];
    // If the target bit is in a |0>/|1> eigenstate, this gate has no effect.
    if (shard.isProbDirty || !((shard.prob < min_norm) || ((ONE_R1 - shard.prob) < min_norm))) {
        CTRLED_CALL_WRAP(
            ApplyControlledSinglePhase(CTRL_P_ARGS), ApplySinglePhase(topLeft, bottomRight, true, target), false);
    }
}

void QUnit::ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex topRight, const complex bottomLeft)
{
    CTRLED_CALL_WRAP(
        ApplyControlledSingleInvert(CTRL_I_ARGS), ApplySingleInvert(topRight, bottomLeft, true, target), false);
}

void QUnit::ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    CTRLED_CALL_WRAP(
        ApplyAntiControlledSinglePhase(CTRL_P_ARGS), ApplySinglePhase(topLeft, bottomRight, true, target), true);
}

void QUnit::ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    CTRLED_CALL_WRAP(
        ApplyAntiControlledSingleInvert(CTRL_I_ARGS), ApplySingleInvert(topRight, bottomLeft, true, target), true);
}

void QUnit::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    shards[qubit].isProbDirty = true;
    shards[qubit].isPhaseDirty = true;
    shards[qubit].unit->ApplySingleBit(mtrx, doCalcNorm, shards[qubit].mapped);
}

void QUnit::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    CTRLED_CALL_WRAP(ApplyControlledSingleBit(CTRL_ARGS), ApplySingleBit(mtrx, true, target), false);
}

void QUnit::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    CTRLED_CALL_WRAP(ApplyAntiControlledSingleBit(CTRL_ARGS), ApplySingleBit(mtrx, true, target), true);
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
    bitProb = Prob(controlVec[controlIndex]);                                                                          \
    if (bitProb < min_norm) {                                                                                          \
        if (!anti) {                                                                                                   \
            /* This gate does nothing, so return without applying anything. */                                         \
            return;                                                                                                    \
        }                                                                                                              \
        /* This control has 100% chance to "fire," so don't entangle it. */                                            \
        controlVec.erase(controlVec.begin() + controlIndex);                                                           \
    } else if ((ONE_R1 - bitProb) < min_norm) {                                                                        \
        if (anti) {                                                                                                    \
            /* This gate does nothing, so return without applying anything. */                                         \
            return;                                                                                                    \
        }                                                                                                              \
        /* This control has 100% chance to "fire," so don't entangle it. */                                            \
        controlVec.erase(controlVec.begin() + controlIndex);                                                           \
    } else {                                                                                                           \
        controlIndex++;                                                                                                \
    }

template <typename CF, typename F>
void QUnit::ApplyEitherControlled(const bitLenInt* controls, const bitLenInt& controlLen,
    const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F fn)
{
    bitLenInt i, j;

    // If the controls start entirely separated from the targets, it's probably worth checking to see if the have total
    // or no probability of altering the targets, such that we can still keep them separate.

    std::vector<bitLenInt> controlVec(controlLen);
    std::copy(controls, controls + controlLen, controlVec.begin());
    bitLenInt controlIndex = 0;

    bool isSeparated = true;
    real1 bitProb;
    for (i = 0; i < controlLen; i++) {
        // If the shard's probability is cached, then it's free to check it, so we advance the loop.
        if (!shards[controls[i]].isProbDirty) {
            // This might determine that we can just skip out of the whole gate, in which case it returns this method:
            CHECK_BREAK_AND_TRIM();
        } else {
            controlIndex++;
            for (j = 0; j < targets.size(); j++) {
                // If the shard doesn't have a cached probability, and if it's in the same shard unit as any of the
                // targets, it isn't worth trying the next optimization.
                if (shards[controls[i]].unit == shards[targets[j]].unit) {
                    isSeparated = false;
                    break;
                }
            }
        }
    }

    bitLenInt controlsLeft = controlVec.size();
    if (isSeparated) {
        // The controls are entirely separated from the targets already, in this branch. If the probability of a change
        // in state from this gate is 0 or 1, we can just act the gate or skip it, without entangling the bits further.
        controlIndex = 0;
        for (i = 0; i < controlsLeft; i++) {
            // This might determine that we can just skip out of the whole gate, in which case it returns this method:
            CHECK_BREAK_AND_TRIM();
        }
        if (controlVec.size() == 0) {
            // Here, the gate is guaranteed to act as if it wasn't controlled, so we apply the gate without controls,
            // avoiding an entangled representation.
            fn();

            return;
        }
    }

    // If we've made it this far, we have to form the entangled representation and apply the gate.
    std::vector<bitLenInt> allBits(controlVec.size() + targets.size());
    std::copy(controlVec.begin(), controlVec.end(), allBits.begin());
    std::copy(targets.begin(), targets.end(), allBits.begin() + controlVec.size());
    std::sort(allBits.begin(), allBits.end());

    std::vector<bitLenInt*> ebits(controlVec.size() + targets.size());
    for (i = 0; i < (controlVec.size() + targets.size()); i++) {
        ebits[i] = &allBits[i];
    }

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    std::vector<bitLenInt> controlsMapped(controlVec.size() == 0 ? 1 : controlVec.size());
    for (i = 0; i < controlVec.size(); i++) {
        controlsMapped[i] = shards[controlVec[i]].mapped;
        shards[controlVec[i]].isPhaseDirty = true;
    }

    cfn(shards[targets[0]].unit, controlsMapped);

    for (i = 0; i < targets.size(); i++) {
        shards[targets[i]].isProbDirty = true;
        shards[targets[i]].isPhaseDirty = true;
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

void QUnit::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLAND(qInputStart + i, cBit, outputStart + i);
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

void QUnit::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLOR(qInputStart + i, cBit, outputStart + i);
    }
}

void QUnit::XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    for (bitLenInt i = 0; i < length; i++) {
        XOR(inputStart1 + i, inputStart2 + i, outputStart + i);
    }
}

void QUnit::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLXOR(qInputStart + i, cBit, outputStart + i);
    }
}

bool QUnit::CArithmeticOptimize(
    bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>* controlVec)
{
    for (auto i = 0; i < controlLen; i++) {
        // If any control has a cached zero probability, this gate will do nothing, and we can avoid basically all
        // overhead.
        if (!shards[controls[i]].isProbDirty && (Prob(controls[i]) < min_norm)) {
            return true;
        }
    }

    // Otherwise, we have to entangle the register.
    EntangleRange(start, length);

    controlVec->resize(controlLen);
    std::copy(controls, controls + controlLen, controlVec->begin());
    bitLenInt controlIndex = 0;

    for (auto i = 0; i < controlLen; i++) {
        if (shards[controls[i]].isProbDirty && (shards[controls[i]].unit == shards[start].unit)) {
            continue;
        }

        real1 prob = Prob(controls[i]);
        if (prob < min_norm) {
            // If any control has zero probability, this gate will do nothing.
            return true;
        } else if (((ONE_R1 - prob) < min_norm) && (shards[controls[i]].unit != shards[start].unit)) {
            // If any control has full probability, we can avoid entangling it.
            controlVec->erase(controlVec->begin() + controlIndex);
        } else {
            controlIndex++;
        }
    }

    return false;
}

void QUnit::CINT(
    CINTFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (CArithmeticOptimize(start, length, controls, controlLen, &controlVec)) {
        // We've determined we can skip the entire gate.
        return;
    }

    // Otherwise, we have to "dirty" the register.
    std::vector<bitLenInt> bits(controlVec.size() + 1);
    for (bitLenInt i = 0; i < controlVec.size(); i++) {
        bits[i] = controlVec[i];
    }
    bits[controlVec.size()] = start;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(controlVec.size() + 1);
    for (bitLenInt i = 0; i < (controlVec.size() + 1); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    std::vector<bitLenInt> controlsMapped(controlVec.size() == 0 ? 1 : controlVec.size());
    for (bitLenInt i = 0; i < controlVec.size(); i++) {
        controlsMapped[i] = shards[controlVec[i]].mapped;
        shards[controlVec[i]].isPhaseDirty = true;
    }

    ((*unit).*fn)(toMod, shards[start].mapped, length, &(controlsMapped[0]), controlVec.size());

    DirtyShardRange(start, length);
}

void QUnit::CINC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        INC(toMod, start, length);
        return;
    }

    CINT(&QInterface::CINC, toMod, start, length, controls, controlLen);
}

void QUnit::CDEC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DEC(toMod, start, length);
        return;
    }

    CINT(&QInterface::CDEC, toMod, start, length, controls, controlLen);
}

/// Collapse the carry bit in an optimal way, before carry arithmetic.
void QUnit::CollapseCarry(bitLenInt flagIndex, bitLenInt start, bitLenInt length)
{
    // Measure the carry flag.
    // Don't separate the flag just to entangle it again, if it's in the same unit.
    QInterfacePtr flagUnit = shards[flagIndex].unit;
    bool isFlagEntangled = false;
    if (flagUnit->GetQubitCount() > 1) {
        for (bitLenInt i = 0; i < length; i++) {
            if (flagUnit == shards[start + i].unit) {
                isFlagEntangled = true;
                break;
            }
        }
    }
    if (isFlagEntangled) {
        flagUnit->M(shards[flagIndex].mapped);
    } else {
        M(flagIndex);
    }
}

void QUnit::INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CollapseCarry(flagIndex, start, length);

    /* Make sure the flag bit is entangled in the same QU. */
    EntangleRange(start, length);

    std::vector<bitLenInt> bits(2);
    bits[0] = start;
    bits[1] = flagIndex;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(2);
    ebits[0] = &bits[0];
    ebits[1] = &bits[1];

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    ((*unit).*fn)(toMod, shards[start].mapped, length, shards[flagIndex].mapped);

    DirtyShardRange(start, length);
    shards[flagIndex].isProbDirty = true;
    shards[flagIndex].isPhaseDirty = true;
}

void QUnit::INCxx(
    INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index)
{
    /*
     * Overflow flag should not be measured, however the carry flag still needs
     * to be measured.
     */
    CollapseCarry(flag2Index, start, length);

    /* Make sure the flag bits are entangled in the same QU. */
    EntangleRange(start, length);
    std::vector<bitLenInt> bits(3);
    bits[0] = start;
    bits[1] = flag1Index;
    bits[2] = flag2Index;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(3);
    ebits[0] = &bits[0];
    ebits[1] = &bits[1];
    ebits[2] = &bits[2];

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

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

    bitCapInt lengthPower = 1U << length;
    bitCapInt signMask = 1U << (length - 1U);
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
        outInt &= (lengthPower - 1U);
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

void QUnit::INC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(start, length)) {
        SetReg(start, length, GetCachedPermutation(start, length) + toMod);
        return;
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(start, length);
    shards[start].unit->INC(toMod, shards[start].mapped, length);
    DirtyShardRange(start, length);
}

void QUnit::INCC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (INTCOptimize(toMod, start, length, true, carryIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCx(&QInterface::INCC, toMod, start, length, carryIndex);
}

void QUnit::INCS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (INTSOptimize(toMod, start, length, true, overflowIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCx(&QInterface::INCS, toMod, start, length, overflowIndex);
}

void QUnit::INCSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (INTSCOptimize(toMod, start, length, true, carryIndex, overflowIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCxx(&QInterface::INCSC, toMod, start, length, overflowIndex, carryIndex);
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

void QUnit::DEC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(start, length)) {
        SetReg(start, length, GetCachedPermutation(start, length) - toMod);
        return;
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(start, length);
    shards[start].unit->DEC(toMod, shards[start].mapped, length);
    DirtyShardRange(start, length);
}

void QUnit::DECC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (INTCOptimize(toMod, start, length, false, carryIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCx(&QInterface::DECC, toMod, start, length, carryIndex);
}

void QUnit::DECS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (INTSOptimize(toMod, start, length, false, overflowIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCx(&QInterface::DECS, toMod, start, length, overflowIndex);
}

void QUnit::DECSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    // Keep the bits separate, if cheap to do so:
    if (INTSCOptimize(toMod, start, length, false, carryIndex, overflowIndex)) {
        return;
    }

    // Otherwise, form the potentially entangled representation:
    INCxx(&QInterface::DECSC, toMod, start, length, overflowIndex, carryIndex);
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
    if (toMul == 0) {
        SetReg(inOutStart, length, 0U);
        SetReg(carryStart, length, 0U);
        return;
    } else if (toMul == 1) {
        SetReg(carryStart, length, 0U);
        return;
    }

    if (CheckBitsPermutation(inOutStart, length)) {
        bitCapInt lengthMask = (1U << length) - 1U;
        bitCapInt res = GetCachedPermutation(inOutStart, length) * toMul;
        SetReg(inOutStart, length, res & lengthMask);
        SetReg(carryStart, length, (res >> length) & lengthMask);
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
    if (toDiv == 1) {
        return;
    }

    if (CheckBitsPermutation(inOutStart, length) && CheckBitsPermutation(carryStart, length)) {
        bitCapInt lengthMask = (1U << length) - 1U;
        bitCapInt origRes =
            GetCachedPermutation(inOutStart, length) | (GetCachedPermutation(carryStart, length) << length);
        bitCapInt res = origRes / toDiv;
        if (origRes == (res * toDiv)) {
            SetReg(inOutStart, length, res & lengthMask);
            SetReg(carryStart, length, (res >> length) & lengthMask);
        }
        return;
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->DIV(toDiv, shards[inOutStart].mapped, shards[carryStart].mapped, length);
    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);
}

void QUnit::MULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    // Inexpensive edge case
    if (toMod == 0) {
        SetReg(outStart, length, 0);
        return;
    }

    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(inStart, length)) {
        bitCapInt res = (GetCachedPermutation(inStart, length) * toMod) % modN;
        SetReg(outStart, length, res);
        return;
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inStart, length, outStart, length);
    shards[inStart].unit->MULModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    DirtyShardRange(outStart, length);
}

void QUnit::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(inStart, length)) {
        bitCapInt res = intPow(toMod, GetCachedPermutation(inStart, length)) % modN;
        SetReg(outStart, length, res);
        return;
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(inStart, length, outStart, length);
    shards[inStart].unit->POWModNOut(toMod, modN, shards[inStart].mapped, shards[outStart].mapped, length);
    DirtyShardRange(outStart, length);
}

QInterfacePtr QUnit::CMULEntangle(std::vector<bitLenInt> controlVec, bitLenInt start, bitCapInt carryStart,
    bitLenInt length, std::vector<bitLenInt>* controlsMapped)
{
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

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

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
    if (CArithmeticOptimize(start, length, controls, controlLen, &controlVec)) {
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
    bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    // Try to optimize away the whole gate, or as many controls as is opportune.
    std::vector<bitLenInt> controlVec;
    if (CArithmeticOptimize(start, length, controls, controlLen, &controlVec)) {
        // We've determined we can skip the entire operation:
        return;
    }

    std::vector<bitLenInt> controlsMapped;
    QInterfacePtr unit = CMULEntangle(controlVec, start, carryStart, length, &controlsMapped);

    ((*unit).*fn)(
        toMod, modN, shards[start].mapped, shards[carryStart].mapped, length, &(controlsMapped[0]), controlVec.size());
}

void QUnit::CMUL(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MUL(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QInterface::CMUL, toMod, start, carryStart, length, controls, controlLen);
}

void QUnit::CDIV(
    bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DIV(toMod, start, carryStart, length);
        return;
    }

    CMULx(&QInterface::CDIV, toMod, start, carryStart, length, controls, controlLen);
}

void QUnit::CMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MULModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CMULModx(&QInterface::CMULModNOut, toMod, modN, inStart, outStart, length, controls, controlLen);
}

void QUnit::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CMULModx(&QInterface::CPOWModNOut, toMod, modN, inStart, outStart, length, controls, controlLen);
}

void QUnit::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(start, length)) {
        if (GetCachedPermutation(start, length) == 0) {
            // This has no physical effect, but we do it to respect direct simulator check of amplitudes:
            shards[start].unit->PhaseFlip();
        }
        return;
    }

    // Otherwise, form the potentially entangled representation:
    EntangleRange(start, length);
    shards[start].unit->ZeroPhaseFlip(shards[start].mapped, length);
    DirtyShardRange(start, length);
}

void QUnit::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    // Keep the bits separate, if cheap to do so:
    if (CheckBitsPermutation(start, length)) {
        if (GetCachedPermutation(start, length) < greaterPerm) {
            // This has no physical effect, but we do it to respect direct simulator check of amplitudes:
            shards[start].unit->PhaseFlip();
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

    std::vector<bitLenInt> bits(2);
    bits[0] = start;
    bits[1] = flagIndex;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(2);
    ebits[0] = &bits[0];
    ebits[1] = &bits[1];

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    unit->CPhaseFlipIfLess(greaterPerm, shards[start].mapped, length, shards[flagIndex].mapped);

    DirtyShardRange(start, length);
    shards[flagIndex].isProbDirty = true;
    shards[flagIndex].isPhaseDirty = true;
}

void QUnit::PhaseFlip() { shards[0].unit->PhaseFlip(); }

bitCapInt QUnit::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    EntangleRange(indexStart, indexLength, valueStart, valueLength);

    bitCapInt toRet = shards[indexStart].unit->IndexedLDA(
        shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);

    return toRet;
}

bitCapInt QUnit::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
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
    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    bitCapInt toRet = shards[indexStart].unit->IndexedSBC(shards[indexStart].mapped, indexLength,
        shards[valueStart].mapped, valueLength, shards[carryIndex].mapped, values);

    DirtyShardRangePhase(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].isProbDirty = true;
    shards[carryIndex].isPhaseDirty = true;

    return toRet;
}

void QUnit::UpdateRunningNorm()
{
    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (find(units.begin(), units.end(), toFind) == units.end()) {
            units.push_back(toFind);
            toFind->UpdateRunningNorm();
        }
    }
}

void QUnit::NormalizeState(real1 nrm)
{
    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (find(units.begin(), units.end(), toFind) == units.end()) {
            units.push_back(toFind);
            toFind->NormalizeState(nrm);
        }
    }
}

void QUnit::Finish()
{
    if (shards[0].unit == NULL) {
        // Uninitialized or already freed
        return;
    }

    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (find(units.begin(), units.end(), toFind) == units.end()) {
            units.push_back(toFind);
            toFind->Finish();
        }
    }
}

bool QUnit::isFinished()
{
    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (find(units.begin(), units.end(), toFind) == units.end()) {
            units.push_back(toFind);
            if (!(toFind->isFinished())) {
                return false;
            }
        }
    }

    return true;
}

bool QUnit::ApproxCompare(QUnitPtr toCompare)
{
    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        return false;
    }

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
    QUnitPtr copyPtr = std::make_shared<QUnit>(engine, subengine, qubitCount, 0, rand_generator,
        complex(ONE_R1, ZERO_R1), doNormalize, randGlobalPhase, useHostRam);

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

        copyPtr->shards[i].unit = dupeEngines[engineIndex];
        copyPtr->shards[i].mapped = shards[i].mapped;
        copyPtr->shards[i].prob = shards[i].prob;
        copyPtr->shards[i].isProbDirty = shards[i].isProbDirty;
        copyPtr->shards[i].phase = shards[i].phase;
        copyPtr->shards[i].isPhaseDirty = shards[i].isPhaseDirty;
    }

    return copyPtr;
}

} // namespace Qrack
