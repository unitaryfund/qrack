//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
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
    bool doNorm, bool randomGlobalPhase, bool useHostMem)
    : QUnit(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem)
{
    // Intentionally left blank
}

QUnit::QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem)
    : QInterface(qBitCount, rgp, doNorm)
    , engine(eng)
    , subengine(subEng)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , randGlobalPhase(randomGlobalPhase)
    , useHostRam(useHostMem)
{
    shards.resize(qBitCount);

    bool bitState;
    for (bitLenInt i = 0; i < qBitCount; i++) {
        bitState = ((1 << i) & initState) >> i;
        shards[i].unit = CreateQuantumInterface(engine, subengine, 1, bitState ? 1 : 0, rand_generator, phaseFactor,
            doNormalize, randGlobalPhase, useHostRam);
        shards[i].mapped = 0;
        shards[i].prob = bitState ? ONE_R1 : ZERO_R1;
        shards[i].isProbDirty = false;
    }
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool bitState;

    Finish();

    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((1 << i) & perm) >> i;
        shards[i].unit = CreateQuantumInterface(engine, subengine, 1, ((1 << i) & perm) >> i, rand_generator, phaseFac,
            doNormalize, randGlobalPhase, useHostRam);
        shards[i].mapped = 0;
        shards[i].prob = bitState ? ONE_R1 : ZERO_R1;
        shards[i].isProbDirty = false;
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
        if (otherUnits.find(otherShard.unit) == otherUnits.end()) {
            otherUnits[otherShard.unit] =
                CreateQuantumInterface(engine, subengine, 1, 0, rand_generator, phaseFactor, doNormalize, useHostRam);
            otherUnits[otherShard.unit]->CopyState(otherShard.unit);
        }
        shard.unit = otherUnits[otherShard.unit];
        shards.push_back(shard);
    }
}

void QUnit::CopyState(QInterfacePtr orig)
{
    QInterfacePtr unit = CreateQuantumInterface(
        engine, subengine, orig->GetQubitCount(), 0, rand_generator, phaseFactor, doNormalize, useHostRam);
    unit->CopyState(orig);

    SetQubitCount(orig->GetQubitCount());
    shards.clear();

    /* Set up the shards to refer to the new unit. */
    for (bitLenInt i = 0; i < (orig->GetQubitCount()); i++) {
        QEngineShard shard;
        shard.unit = unit;
        shard.mapped = i;
        shard.isProbDirty = true;
        shards.push_back(shard);
    }
}

void QUnit::SetQuantumState(complex* inputState)
{
    auto unit =
        CreateQuantumInterface(engine, subengine, qubitCount, 0, rand_generator, phaseFactor, doNormalize, useHostRam);
    unit->SetQuantumState(inputState);

    int idx = 0;
    for (auto&& shard : shards) {
        shard.unit = unit;
        shard.mapped = idx++;
        shard.isProbDirty = true;
    }
}

void QUnit::GetQuantumState(complex* outputState)
{
    QUnit qUnitCopy(engine, subengine, 1, 0);
    qUnitCopy.CopyState((QUnit*)this);
    qUnitCopy.EntangleAll()->GetQuantumState(outputState);
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

    return result;
}

/*
 * Append QInterface to the end of the unit.
 */
void QUnit::Compose(QUnitPtr toCopy, bool isMid, bitLenInt start)
{
    bitLenInt oQubitCount = toCopy->GetQubitCount();
    bitLenInt oldCount = qubitCount;

    /* Increase the number of bits in this object. */
    SetQubitCount(qubitCount + oQubitCount);

    /* Create a clone of the quantum state in toCopy. */
    QUnitPtr clone(toCopy);

    /* Update shards to reference the cloned state. */
    bitLenInt j;
    for (bitLenInt i = 0; i < clone->GetQubitCount(); i++) {
        j = i + oldCount;
        shards[j].unit = clone->shards[i].unit;
        shards[j].mapped = clone->shards[i].mapped;
        shards[j].prob = clone->shards[i].prob;
        shards[j].isProbDirty = clone->shards[i].isProbDirty;
    }

    if (isMid) {
        ROL(oQubitCount, start, qubitCount - start);
    }
}

bitLenInt QUnit::Compose(QUnitPtr toCopy)
{
    bitLenInt oldCount = qubitCount;
    Compose(toCopy, false, 0);
    return oldCount;
}

/*
 * Append QInterface in the middle of QUnit.
 */
bitLenInt QUnit::Compose(QUnitPtr toCopy, bitLenInt start)
{
    Compose(toCopy, true, start);
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
        complex(ONE_R1, ZERO_R1), doNormalize, randGlobalPhase, useHostRam);

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

    return ClampProb(result);
}

bool QUnit::ForceM(bitLenInt qubit, bool res, bool doForce)
{
    bool result = shards[qubit].unit->ForceM(shards[qubit].mapped, res, doForce);

    shards[qubit].prob = result ? ONE_R1 : ZERO_R1;
    shards[qubit].isProbDirty = false;

    QInterfacePtr unit = shards[qubit].unit;
    bitLenInt mapped = shards[qubit].mapped;

    if (unit->GetQubitCount() == 1) {
        /* If we're keeping the bits, and they're already in their own unit, there's nothing to do. */
        return result;
    }

    QInterfacePtr dest = CreateQuantumInterface(
        engine, subengine, 1, result ? 1 : 0, rand_generator, phaseFactor, doNormalize, useHostRam);
    unit->Dispose(mapped, 1);

    /* Update the mappings. */
    shards[qubit].unit = dest;
    shards[qubit].mapped = 0;
    for (auto&& shard : shards) {
        if (shard.unit == unit && shard.mapped > mapped) {
            shard.mapped--;
        }
    }

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

    EntangleAndCallMember(PTR2(SqrtSwap), qubit1, qubit2);
}

void QUnit::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    shards[qubit1].isProbDirty = true;
    shards[qubit2].isProbDirty = true;

    EntangleAndCallMember(PTR2(ISqrtSwap), qubit1, qubit2);
}

bool QUnit::DoesOperatorPhaseShift(const complex* mtrx)
{
    bool doesShift = false;
    real1 phase = -M_PI * 2;
    for (int i = 0; i < 4; i++) {
        if (norm(mtrx[i]) > min_norm) {
            if (phase < -M_PI) {
                phase = arg(mtrx[i]);
                continue;
            }

            real1 diff = arg(mtrx[i]) - phase;
            if (diff < ZERO_R1) {
                diff = -diff;
            }
            if (diff > M_PI) {
                diff = (2 * M_PI) - diff;
            }
            if (diff > min_norm) {
                doesShift = true;
                break;
            }
        }
    }

    return doesShift;
}

void QUnit::UniformlyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const complex* mtrxs)
{
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
    }

    shards[qubitIndex].isProbDirty = true;
    unit->UniformlyControlledSingleBit(mappedControls, controlLen, shards[qubitIndex].mapped, mtrxs);

    delete[] mappedControls;
}

void QUnit::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    shards[qubit].isProbDirty = true;
    shards[qubit].unit->ApplySingleBit(mtrx, doCalcNorm, shards[qubit].mapped);
}

void QUnit::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    ApplyEitherControlled(controls, controlLen, { target }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->ApplyControlledSingleBit(mappedControls, controlLen, shards[target].mapped, mtrx);
        },
        [&]() { ApplySingleBit(mtrx, true, target); });
}

void QUnit::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    ApplyEitherControlled(controls, controlLen, { target }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->ApplyAntiControlledSingleBit(mappedControls, controlLen, shards[target].mapped, mtrx);
        },
        [&]() { ApplySingleBit(mtrx, true, target); });
}

void QUnit::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->CSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
        },
        [&]() { Swap(qubit1, qubit2); });
}

void QUnit::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->AntiCSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
        },
        [&]() { Swap(qubit1, qubit2); });
}

void QUnit::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->CSqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
        },
        [&]() { SqrtSwap(qubit1, qubit2); });
}

void QUnit::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->AntiCSqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
        },
        [&]() { SqrtSwap(qubit1, qubit2); });
}

void QUnit::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->CISqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
        },
        [&]() { ISqrtSwap(qubit1, qubit2); });
}

void QUnit::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->AntiCISqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
        },
        [&]() { ISqrtSwap(qubit1, qubit2); });
}

template <typename CF, typename F>
void QUnit::ApplyEitherControlled(const bitLenInt* controls, const bitLenInt& controlLen,
    const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F fn)
{
    int i, j;

    // If the controls start entirely separated from the targets, it's probably worth checking to see if the have total
    // or no probability of altering the targets, such that we can still keep them separate.

    bool isSeparated = true;
    for (i = 0; i < controlLen; i++) {
        // If the shard's probability is cached, then it's free to check it, so we advance the loop.
        if (!shards[controls[i]].isProbDirty) {
            continue;
        }
        for (j = 0; j < (int)targets.size(); j++) {
            // If the shard doesn't have a cached probability, and if it's in the same shard unit as any of the targets,
            // it isn't worth trying the next optimization.
            if (shards[controls[i]].unit == shards[targets[j]].unit) {
                isSeparated = false;
                break;
            }
        }
        if (!isSeparated) {
            break;
        }
    }

    if (isSeparated) {
        // The controls are entirely separated from the targets already, in this branch. If the probability of a change
        // in state from this gate is 0 or 1, we can just act the gate or skip it, without entangling the bits further.
        real1 prob = ONE_R1;
        real1 bitProb;
        for (i = 0; i < controlLen; i++) {
            bitProb = Prob(controls[i]);

            if (anti) {
                prob *= ONE_R1 - bitProb;
            } else {
                prob *= bitProb;
            }
            if (prob < min_norm) {
                break;
            }
        }
        if (prob < min_norm) {
            // Here, the gate is guaranteed not to have any effect, so we skip it.
            return;
        } else if ((ONE_R1 - prob) < min_norm) {
            // Here, the gate is guaranteed to act as if it wasn't controlled, so we apply the gate without controls,
            // avoiding an entangled representation.
            fn();

            for (i = 0; i < (bitLenInt)targets.size(); i++) {
                shards[targets[i]].isProbDirty = true;
            }

            return;
        }
    }

    // If we've made it this far, we have to form the entangled representation and apply the gate.
    std::vector<bitLenInt> allBits(controlLen + targets.size());
    std::copy(controls, controls + controlLen, allBits.begin());
    std::copy(targets.begin(), targets.end(), allBits.begin() + controlLen);
    std::sort(allBits.begin(), allBits.end());

    std::vector<bitLenInt*> ebits(controlLen + targets.size());
    for (i = 0; i < (int)(controlLen + targets.size()); i++) {
        ebits[i] = &allBits[i];
    }

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    bitLenInt* controlsMapped = new bitLenInt[controlLen];
    for (i = 0; i < controlLen; i++) {
        controlsMapped[i] = shards[controls[i]].mapped;
    }

    cfn(unit, controlsMapped);

    for (i = 0; i < (bitLenInt)targets.size(); i++) {
        shards[targets[i]].isProbDirty = true;
    }

    delete[] controlsMapped;
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

void QUnit::INC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    DirtyShardRange(start, length);

    EntangleRange(start, length);
    shards[start].unit->INC(toMod, shards[start].mapped, length);
}

void QUnit::CINT(
    CINTFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    DirtyShardRange(start, length);
    DirtyShardIndexArray(controls, controlLen);

    EntangleRange(start, length);
    std::vector<bitLenInt> bits(controlLen + 1);
    for (auto i = 0; i < controlLen; i++) {
        bits[i] = controls[i];
    }
    bits[controlLen] = start;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(controlLen + 1);
    for (auto i = 0; i < (controlLen + 1); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    bitLenInt* controlsMapped = new bitLenInt[controlLen];
    for (auto i = 0; i < controlLen; i++) {
        controlsMapped[i] = shards[controls[i]].mapped;
    }

    ((*unit).*fn)(toMod, shards[start].mapped, length, controlsMapped, controlLen);

    delete[] controlsMapped;
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

void QUnit::INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    /*
     * FUTURE: If start[length] and carry are already in the same QE, then it
     * doesn't make sense to Decompose and re-entangle them.
     */
    M(flagIndex);

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
}

void QUnit::INCxx(
    INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index)
{
    /*
     * FUTURE: If start[length] and carry are already in the same QE, then it
     * doesn't make sense to Decompose and re-entangle them.
     */

    /*
     * Overflow flag should not be measured, however the carry flag still needs
     * to be measured.
     */
    M(flag2Index);

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
}

void QUnit::INCC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::INCC, toMod, start, length, carryIndex);
}

void QUnit::INCS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INCx(&QInterface::INCS, toMod, start, length, overflowIndex);
}

void QUnit::INCSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INCxx(&QInterface::INCSC, toMod, start, length, overflowIndex, carryIndex);
}

void QUnit::INCSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::INCSC, toMod, start, length, carryIndex);
}

void QUnit::INCBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    DirtyShardRange(start, length);

    EntangleRange(start, length);
    shards[start].unit->INCBCD(toMod, shards[start].mapped, length);
}

void QUnit::INCBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::INCBCDC, toMod, start, length, carryIndex);
}

void QUnit::DEC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    DirtyShardRange(start, length);

    EntangleRange(start, length);
    shards[start].unit->DEC(toMod, shards[start].mapped, length);
}

void QUnit::DECC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::DECC, toMod, start, length, carryIndex);
}

void QUnit::DECS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INCx(&QInterface::DECS, toMod, start, length, overflowIndex);
}

void QUnit::DECSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INCxx(&QInterface::DECSC, toMod, start, length, overflowIndex, carryIndex);
}

void QUnit::DECSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::DECSC, toMod, start, length, carryIndex);
}

void QUnit::DECBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    DirtyShardRange(start, length);

    EntangleRange(start, length);
    shards[start].unit->DECBCD(toMod, shards[start].mapped, length);
}

void QUnit::DECBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::DECBCDC, toMod, start, length, carryIndex);
}

void QUnit::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);

    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->MUL(toMul, shards[inOutStart].mapped, shards[carryStart].mapped, length);
}

void QUnit::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    DirtyShardRange(inOutStart, length);
    DirtyShardRange(carryStart, length);

    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->DIV(toDiv, shards[inOutStart].mapped, shards[carryStart].mapped, length);
}

void QUnit::CMULx(CMULFn fn, bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    EntangleRange(start, length);
    EntangleRange(carryStart, length);
    std::vector<bitLenInt> bits(controlLen + 2);
    for (auto i = 0; i < controlLen; i++) {
        bits[i] = controls[i];
    }
    bits[controlLen] = start;
    bits[controlLen + 1] = carryStart;
    std::sort(bits.begin(), bits.end());

    std::vector<bitLenInt*> ebits(controlLen + 2);
    for (auto i = 0; i < (controlLen + 2); i++) {
        ebits[i] = &bits[i];
    }

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    bitLenInt* controlsMapped = new bitLenInt[controlLen];
    for (auto i = 0; i < controlLen; i++) {
        controlsMapped[i] = shards[controls[i]].mapped;
    }

    ((*unit).*fn)(toMod, shards[start].mapped, shards[carryStart].mapped, length, controlsMapped, controlLen);

    delete[] controlsMapped;

    DirtyShardRange(start, length);
    DirtyShardRange(carryStart, length);
    DirtyShardIndexArray(controls, controlLen);
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

void QUnit::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    DirtyShardRange(start, length);

    EntangleRange(start, length);
    shards[start].unit->ZeroPhaseFlip(shards[start].mapped, length);
}

void QUnit::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    DirtyShardRange(start, length);

    EntangleRange(start, length);
    shards[start].unit->PhaseFlipIfLess(greaterPerm, shards[start].mapped, length);
}

void QUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    DirtyShardRange(start, length);
    shards[flagIndex].isProbDirty = true;

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
}

void QUnit::PhaseFlip() { shards[0].unit->PhaseFlip(); }

bitCapInt QUnit::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    DirtyShardRange(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);

    EntangleRange(indexStart, indexLength, valueStart, valueLength);

    return shards[indexStart].unit->IndexedLDA(
        shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, values);
}

bitCapInt QUnit::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    DirtyShardRange(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].isProbDirty = true;

    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    return shards[indexStart].unit->IndexedADC(shards[indexStart].mapped, indexLength, shards[valueStart].mapped,
        valueLength, shards[carryIndex].mapped, values);
}

bitCapInt QUnit::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    DirtyShardRange(indexStart, indexLength);
    DirtyShardRange(valueStart, valueLength);
    shards[carryIndex].isProbDirty = true;

    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    return shards[indexStart].unit->IndexedSBC(shards[indexStart].mapped, indexLength, shards[valueStart].mapped,
        valueLength, shards[carryIndex].mapped, values);
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

void QUnit::Finish()
{
    std::vector<QInterfacePtr> units;
    for (bitLenInt i = 0; i < shards.size(); i++) {
        QInterfacePtr toFind = shards[i].unit;
        if (find(units.begin(), units.end(), toFind) == units.end()) {
            units.push_back(toFind);
            toFind->Finish();
        }
    }
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
    }

    return copyPtr;
}

} // namespace Qrack
