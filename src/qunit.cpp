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

QUnit::QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState,
    std::shared_ptr<std::default_random_engine> rgp, complex phaseFac, bool doNorm, bool useHostMem)
    : QUnit(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, useHostMem)
{
    // Intentionally left blank
}

QUnit::QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState,
    std::shared_ptr<std::default_random_engine> rgp, complex phaseFac, bool doNorm, bool useHostMem)
    : QInterface(qBitCount, rgp, doNorm)
    , engine(eng)
    , subengine(subEng)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
{
    shards.resize(qBitCount);

    for (bitLenInt i = 0; i < qBitCount; i++) {
        shards[i].unit = CreateQuantumInterface(
            engine, subengine, 1, ((1 << i) & initState) >> i, rand_generator, phaseFactor, doNormalize, useHostRam);
        shards[i].mapped = 0;
        shards[i].isPhaseDirty = false;
    }
}

void QUnit::SetPermutation(bitCapInt perm, complex phaseFac)
{
    Finish();
    for (bitLenInt i = 0; i < qubitCount; i++) {
        shards[i].unit = CreateQuantumInterface(
            engine, subengine, 1, ((1 << i) & perm) >> i, rand_generator, phaseFac, doNormalize, useHostRam);
        shards[i].mapped = 0;
        shards[i].isPhaseDirty = false;
    }
}

void QUnit::CopyState(QUnitPtr orig) { CopyState(orig.get()); }

// protected method
void QUnit::CopyState(QUnit* orig)
{
    Finish();

    SetQubitCount(orig->GetQubitCount());
    shards.clear();

    /* Set up the shards to refer to the new unit. */
    std::map<QInterfacePtr, QInterfacePtr> otherUnits;
    for (auto&& otherShard : orig->shards) {
        QEngineShard shard;
        shard.mapped = otherShard.mapped;
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
    Finish();

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
        shards.push_back(shard);
    }
}

void QUnit::SetQuantumState(complex* inputState)
{
    Finish();

    auto unit =
        CreateQuantumInterface(engine, subengine, qubitCount, 0, rand_generator, phaseFactor, doNormalize, useHostRam);
    unit->SetQuantumState(inputState);

    int idx = 0;
    for (auto&& shard : shards) {
        shard.unit = unit;
        shard.mapped = idx++;
    }

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (!TrySeparate({ i })) {
            shards[i].isPhaseDirty = true;
        }
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
bitLenInt QUnit::Cohere(QInterfacePtr toCopy)
{
    bitLenInt oldCount = qubitCount;

    /* Increase the number of bits in this object. */
    SetQubitCount(qubitCount + toCopy->GetQubitCount());

    /* Create a clone of the quantum state in toCopy. */
    QInterfacePtr clone(toCopy);

    /* Update shards to reference the cloned state. */
    for (bitLenInt i = 0; i < clone->GetQubitCount(); i++) {
        shards[i + oldCount].unit = clone;
        shards[i + oldCount].mapped = i;
    }

    return oldCount;
}

bool QUnit::Detach(bitLenInt start, bitLenInt length, QInterfacePtr dest, bool checkIfSeparable)
{
    /* TODO: This method should compose the bits for the destination without cohering the length first */

    if (length > 1) {
        EntangleRange(start, length);
        OrderContiguous(shards[start].unit);
    }

    QInterfacePtr unit = shards[start].unit;
    bitLenInt mapped = shards[start].mapped;
    bitLenInt unitLength = unit->GetQubitCount();

    if (dest && unit->GetQubitCount() > length) {
        if (checkIfSeparable) {
            if (!(unit->TryDecohere(mapped, length, dest))) {
                return false;
            }
        } else {
            unit->Decohere(mapped, length, dest);
        }
    } else if (dest) {
        dest->CopyState(unit);
    } else {
        unit->Dispose(mapped, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);

    if (unitLength == length) {
        return true;
    }

    /* Find the rest of the qubits. */
    for (auto&& shard : shards) {
        if (shard.unit == unit && shard.mapped > (mapped + length)) {
            shard.mapped -= length;
        }
    }

    return true;
}

void QUnit::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest) { Detach(start, length, dest, false); }

void QUnit::Dispose(bitLenInt start, bitLenInt length) { Detach(start, length, nullptr, false); }

bool QUnit::TryDecohere(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    return Detach(start, length, dest, true);
}

QInterfacePtr QUnit::EntangleIterator(std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    std::vector<QInterfacePtr> units;
    units.reserve((int)(last - first));

    QInterfacePtr unit1 = shards[**first].unit;
    std::map<QInterfacePtr, bool> found;

    found[unit1] = true;

    /* Walk through all of the supplied bits and create a unique list to cohere. */
    for (auto bit = first + 1; bit != last; ++bit) {
        if (found.find(shards[**bit].unit) == found.end()) {
            found[shards[**bit].unit] = true;
            units.push_back(shards[**bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    if (units.size() != 0) {
        auto&& offsets = unit1->Cohere(units);

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
    for (auto bit = first; bit != last; ++bit) {
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

    /* Walk through all of the supplied bits and create a unique list to cohere. */
    for (bitLenInt bit = 1; bit < qubitCount; bit++) {
        if (found.find(shards[bit].unit) == found.end()) {
            found[shards[bit].unit] = true;
            units.push_back(shards[bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    if (units.size() != 0) {
        auto&& offsets = unit1->QInterface::Cohere(units);

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

bool QUnit::TrySeparate(std::vector<bitLenInt> bits)
{
    bool didSeparate = false;
    QEngineShard shard;
    for (bitLenInt i = 0; i < (bits.size()); i++) {
        bool didSeparateBit = false;
        shard = shards[bits[i]];
        if (shard.unit->GetQubitCount() > 1) {
            QInterfacePtr testBit =
                CreateQuantumInterface(engine, subengine, 1, 0, rand_generator, phaseFactor, doNormalize, useHostRam);

            didSeparateBit = shard.unit->TryDecohere(shard.mapped, 1, testBit);

            if (didSeparateBit) {
                // The bit is separable. Keep the test unit, and update the shard mappings.
                shards[bits[i]].unit = testBit;
                shards[bits[i]].mapped = 0;
                for (auto&& shrd : shards) {
                    if ((shrd.unit == shard.unit) && (shrd.mapped > shard.mapped)) {
                        shrd.mapped--;
                    }
                }

                if (shard.isPhaseDirty) {
                    complex amp0 = testBit->GetAmplitude(0);
                    real1 phase0;
                    if (norm(amp0) < min_norm) {
                        shards[bits[i]].isPhaseDirty = false;
                        continue;
                    } else {
                        phase0 = arg(amp0);
                    }

                    complex amp1 = testBit->GetAmplitude(1);
                    real1 phase1;
                    if (norm(amp1) < min_norm) {
                        shards[bits[i]].isPhaseDirty = false;
                        continue;
                    } else {
                        phase1 = arg(amp0);
                    }

                    real1 phaseDiff = std::abs(phase0 - phase1);
                    if (phaseDiff > M_PI) {
                        phaseDiff = 2 * M_PI - phaseDiff;
                    }
                    if (phaseDiff < min_norm) {
                        shards[bits[i]].isPhaseDirty = false;
                    }
                }
            }
        }
        didSeparate |= didSeparateBit;
    }
    return didSeparate;
}

void QUnit::OrderContiguous(QInterfacePtr unit)
{
    /* Before we call OrderContinguous, when we are cohering lists of shards, we should always proactively sort the
     * order in which we cohere qubits into a single engine. This is a cheap way to reduce the need for costly qubit
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
    return (shard.unit->Prob)(shard.mapped);
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

    return result;
}

bool QUnit::ForceM(bitLenInt qubit, bool res, bool doForce)
{
    shards[qubit].isPhaseDirty = false;

    bool result = shards[qubit].unit->ForceM(shards[qubit].mapped, res, doForce);

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

    for (bitLenInt i = 0; i < length; i++) {
        shards[i + start].unit->SetPermutation((value & (1 << i)) > 0 ? 1 : 0);
    }
}

void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    QEngineShard& shard1 = shards[qubit1];
    QEngineShard& shard2 = shards[qubit2];

    QEngineShard tmp;

    // Swap the bit mapping.
    tmp.mapped = shard1.mapped;
    shard1.mapped = shard2.mapped;
    shard2.mapped = tmp.mapped;

    // Swap the QInterface object.
    tmp.unit = shard1.unit;
    shard1.unit = shard2.unit;
    shard2.unit = tmp.unit;
}

/* Unfortunately, many methods are overloaded, which prevents using just the address-to-member. */
#define PTR3(OP) (void (QInterface::*)(bitLenInt, bitLenInt, bitLenInt))(&QInterface::OP)
#define PTR2(OP) (void (QInterface::*)(bitLenInt, bitLenInt))(&QInterface::OP)
#define PTR1(OP) (void (QInterface::*)(bitLenInt))(&QInterface::OP)
#define PTR2A(OP) (void (QInterface::*)(real1, bitLenInt, bitLenInt))(&QInterface::OP)
#define PTRA(OP) (void (QInterface::*)(real1, bitLenInt))(&QInterface::OP)

void QUnit::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    EntangleAndCallMember(PTR2(SqrtSwap), qubit1, qubit2);
    shards[qubit1].isPhaseDirty = true;
    shards[qubit2].isPhaseDirty = true;
}

void QUnit::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    EntangleAndCallMember(PTR2(ISqrtSwap), qubit1, qubit2);
    shards[qubit1].isPhaseDirty = true;
    shards[qubit2].isPhaseDirty = true;
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

void QUnit::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    // If this operation can induce a superposition of phase, mark the shard "isPhaseDirty." This is necessary to track
    // entanglement.
    if (DoesOperatorPhaseShift(mtrx)) {
        shards[qubit].isPhaseDirty = true;
        // This operation might make a "phase dirty" bit into a "phase clean" bit for entanglement, but we can't detect
        // that, yet.
    }
    shards[qubit].unit->ApplySingleBit(mtrx, doCalcNorm, shards[qubit].mapped);
}

void QUnit::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    ApplyEitherControlled(controls, controlLen, { target }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->ApplyControlledSingleBit(mappedControls, controlLen, shards[target].mapped, mtrx);
            if (DoesOperatorPhaseShift(mtrx)) {
                shards[target].isPhaseDirty = true;
                for (bitLenInt i = 0; i < controlLen; i++) {
                    shards[controls[i]].isPhaseDirty = true;
                }
            }
            return TrySeparate({ target });
        },
        [&]() { ApplySingleBit(mtrx, true, target); });
}

void QUnit::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    ApplyEitherControlled(controls, controlLen, { target }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->ApplyAntiControlledSingleBit(mappedControls, controlLen, shards[target].mapped, mtrx);
            if (DoesOperatorPhaseShift(mtrx)) {
                shards[target].isPhaseDirty = true;
                for (bitLenInt i = 0; i < controlLen; i++) {
                    shards[controls[i]].isPhaseDirty = true;
                }
            }
            return TrySeparate({ target });
        },
        [&]() { ApplySingleBit(mtrx, true, target); });
}

void QUnit::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->CSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
            return TrySeparate({ qubit1, qubit2 });
        },
        [&]() { Swap(qubit1, qubit2); });
}

void QUnit::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->AntiCSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
            return TrySeparate({ qubit1, qubit2 });
        },
        [&]() { Swap(qubit1, qubit2); });
}

void QUnit::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->CSqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
            shards[qubit1].isPhaseDirty = true;
            shards[qubit2].isPhaseDirty = true;
            for (bitLenInt i = 0; i < controlLen; i++) {
                shards[controls[i]].isPhaseDirty = true;
            }
            return TrySeparate({ qubit1, qubit2 });
        },
        [&]() { SqrtSwap(qubit1, qubit2); });
}

void QUnit::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->AntiCSqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
            shards[qubit1].isPhaseDirty = true;
            shards[qubit2].isPhaseDirty = true;
            for (bitLenInt i = 0; i < controlLen; i++) {
                shards[controls[i]].isPhaseDirty = true;
            }
            return TrySeparate({ qubit1, qubit2 });
        },
        [&]() { SqrtSwap(qubit1, qubit2); });
}

void QUnit::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, false,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->CISqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
            shards[qubit1].isPhaseDirty = true;
            shards[qubit2].isPhaseDirty = true;
            for (bitLenInt i = 0; i < controlLen; i++) {
                shards[controls[i]].isPhaseDirty = true;
            }
            return TrySeparate({ qubit1, qubit2 });
        },
        [&]() { ISqrtSwap(qubit1, qubit2); });
}

void QUnit::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    ApplyEitherControlled(controls, controlLen, { qubit1, qubit2 }, true,
        [&](QInterfacePtr unit, bitLenInt* mappedControls) {
            unit->AntiCISqrtSwap(mappedControls, controlLen, shards[qubit1].mapped, shards[qubit2].mapped);
            shards[qubit1].isPhaseDirty = true;
            shards[qubit2].isPhaseDirty = true;
            for (bitLenInt i = 0; i < controlLen; i++) {
                shards[controls[i]].isPhaseDirty = true;
            }
            return TrySeparate({ qubit1, qubit2 });
        },
        [&]() { ISqrtSwap(qubit1, qubit2); });
}

template <typename CF, typename F>
void QUnit::ApplyEitherControlled(const bitLenInt* controls, const bitLenInt& controlLen,
    const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F fn)
{
    int i, j;
    real1 prob = ONE_R1;
    for (i = 0; i < controlLen; i++) {
        if (anti) {
            prob *= ONE_R1 - Prob(controls[i]);
        } else {
            prob *= Prob(controls[i]);
        }
        if (prob <= min_norm) {
            break;
        }
    }
    if (prob <= min_norm) {
        return;
    } else if (min_norm >= (ONE_R1 - prob)) {
        for (i = 0; i < controlLen; i++) {
            if (!shards[controls[i]].isPhaseDirty) {
                ForceM(controls[i], !anti);
            }
        }

        fn();
        return;
    }

    std::vector<bitLenInt> allBits(controlLen + targets.size());
    for (i = 0; i < controlLen; i++) {
        allBits[i] = controls[i];
    }
    for (i = 0; i < (int)targets.size(); i++) {
        allBits[controlLen + i] = targets[i];
    }
    std::sort(allBits.begin() + controlLen, allBits.end());

    std::vector<bitLenInt*> ebits(controlLen + targets.size());
    for (i = 0; i < (int)(controlLen + targets.size()); i++) {
        ebits[i] = &allBits[i];
    }

    QInterfacePtr unit = EntangleIterator(ebits.begin(), ebits.end());

    bitLenInt* controlsMapped = new bitLenInt[controlLen];
    for (i = 0; i < controlLen; i++) {
        controlsMapped[i] = shards[controls[i]].mapped;
    }

    if (!cfn(unit, controlsMapped)) {
        // If "cfn" returns true, it was able to separate bits. This only happens if all permutations are in phase.
        // Otherwise, the phase might be dirty if the controls have dirty phase.
        for (i = 0; i < controlLen; i++) {
            if (shards[controls[i]].isPhaseDirty) {
                for (j = 0; j < (int)targets.size(); j++) {
                    shards[targets[j]].isPhaseDirty = true;
                }
                break;
            }
        }
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
    EntangleRange(start, length);
    shards[start].unit->INC(toMod, shards[start].mapped, length);
}

void QUnit::CINT(
    CINTFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
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
    EntangleRange(start, length);
    shards[start].unit->INCBCD(toMod, shards[start].mapped, length);
}

void QUnit::INCBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::INCBCDC, toMod, start, length, carryIndex);
}

void QUnit::DEC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
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
    EntangleRange(start, length);
    shards[start].unit->DECBCD(toMod, shards[start].mapped, length);
}

void QUnit::DECBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::DECBCDC, toMod, start, length, carryIndex);
}

void QUnit::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    EntangleRange(inOutStart, length, carryStart, length);
    shards[inOutStart].unit->MUL(toMul, shards[inOutStart].mapped, shards[carryStart].mapped, length);
}

void QUnit::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
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
    EntangleRange(start, length);
    shards[start].unit->ZeroPhaseFlip(shards[start].mapped, length);

    for (bitLenInt i = 0; i < length; i++) {
        shards[start + i].isPhaseDirty = true;
    }
}

void QUnit::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    shards[start].unit->PhaseFlipIfLess(greaterPerm, shards[start].mapped, length);

    for (bitLenInt i = 0; i < length; i++) {
        shards[start + i].isPhaseDirty = true;
    }
}

void QUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
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

    for (bitLenInt i = 0; i < length; i++) {
        shards[start + i].isPhaseDirty = true;
    }
    shards[flagIndex].isPhaseDirty = true;
}

void QUnit::PhaseFlip() { shards[0].unit->PhaseFlip(); }

bitCapInt QUnit::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    EntangleRange(indexStart, indexLength, valueStart, valueLength);

    return shards[indexStart].unit->IndexedLDA(
        shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, values);
}

bitCapInt QUnit::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    EntangleRange(indexStart, indexLength, valueStart, valueLength, carryIndex, 1);

    return shards[indexStart].unit->IndexedADC(shards[indexStart].mapped, indexLength, shards[valueStart].mapped,
        valueLength, shards[carryIndex].mapped, values);
}

bitCapInt QUnit::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
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

    QUnit thisCopy(engine, subengine, 1, 0);
    thisCopy.CopyState((QUnit*)this);
    thisCopy.EntangleAll();

    QUnit thatCopy(engine, subengine, 1, 0);
    thatCopy.CopyState(toCompare);
    thatCopy.EntangleAll();

    return thisCopy.shards[0].unit->ApproxCompare(thatCopy.shards[0].unit);
}

} // namespace Qrack
