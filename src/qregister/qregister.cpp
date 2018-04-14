//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "separatedunit.hpp"
#include <iostream>

namespace Qrack {

QRegister::QRegister(CoherentUnitEngine eng, bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac, uint32_t rand_seed) : engine(eng)
{
    rand_generator = std::default_random_engine();
    rand_generator->seed(rand_seed);
    shards.resize(qBitCount);

    if (phaseFac == Complex16(-999.0, -999.0)) {
        double angle = Rand() * 2.0 * M_PI;
        phaseFac = Complex16(cos(angle), sin(angle));
    }

    for (auto shard : shards) {
        shard.unit = CreateCoherentUnit(engine, 1, 0, phaseFac, rand_generator);
        shard.mapped = 0;
    }
}

void QRegister::Decompose(bitLenInt qubit)
{
    std::shared_ptr<QUnit> unit = shards[qubit].unit;
    for (auto shard : shards) {
        if (shard.unit == unit) {
            shard.unit = CreateCoherentUnit(engine, 1, 0, phaseFac, rand_generator);
            shard.unit->SetBit(0, unit->M(shard.mapped)); // Probably wrong, but YWKIM
            shard.mapped = 0;
        }
    }
}

/* XXX Convert this to a variadic template argument function call. */
void QRegister::EntangleAndCall(bitLenInt bit1, bitLenInt bit2, TwoBitCall fn)
{
    std::shared_ptr<QUnit> unit1 = shards[bit1].unit;
    std::shared_ptr<QUnit> unit2 = shards[bit2].unit;

    if (unit1 != unit2) {
        // Not already cohered; create a new unit and merge.
        unit1->Cohere(unit2);

        // Adjust all of the shards that referenced either of the old units.
        for (auto shard : shards) {
            if (shard.unit == unit2) {
                shard.unit = unit1;
                shard.mapped = shard.mapped + unit1->GetQubitCount();
            }
        }
    }

    (unit->*fn)(shards[bit1].mapped, shards[bit2].mapped);
}

void QRegister::EntangleAndCall(bitLenInt bit1, bitLenInt bit2, bitLenInt bit3, ThreeBitCall fn)
{
    std::shared_ptr<QUnit> unit1 = shards[bit1].unit;
    std::shared_ptr<QUnit> unit2 = shards[bit2].unit;
    std::shared_ptr<QUnit> unit3 = shards[bit3].unit;

    if (unit1 != unit2 || unit2 != unit3) {
        // Not already cohered; create a new unit and merge.
        unit1->Cohere(unit2);
        unit1->Cohere(unit3);

        // Adjust all of the shards that referenced either of the old units.
        for (auto shard : shards) {
            if (shard.unit == unit2) {
                shard.unit = unit1;
                shard.mapped = shard.mapped + unit1->GetQubitCount();
            }
            if (shard.unit == unit3) {
                shard.unit = unit1;
                shard.mapped = shard.mapped + unit1->GetQubitCount() + unit2->GetQubitCount();
        }
    }

    (unit->*fn)(shards[bit1].mapped, shards[bit2].mapped, shards[bit3].mapped);
}

double QRegister::Prob(bitLenInt qubit)
{
    QuantumBitShard &shard = shards[qubit];
    return (shard.unit->Prob)(shard.mapped);
}

double QRegister::ProbAll(bitCapInt perm)
{
    double result = 1.0;

    for (auto shard : shards) {
        p = 0;
        //for (auto bit : shards[i].bits) {
        //    p |= perm & (1 << bit) ? (1 << shards[i].bits[bit]) : 0;
        //}
        // XXX: Reconstruct the perm for this particular CU's mapping.
        // result *= Call<&CoherentUnit::ProbAll>(p) << i;
    }

    return result;
}

void QRegister::ProbArray(double* probArray)
{
    for (int bit = 0; bit < shards.length(); bit++) {
        probArray[bit] = Prob(bit);
    }
}

/// Measure a bit
bool QRegister::M(bitLenInt qubit)
{
    QuantumBitShard &shard = shards[qubit];
    bool result = shard.unit->M(shard.mapped);

    /*
     * Decomposes all of the bits in the shard, performing M() on each one and
     * setting each new CU to the appropriate value.
     */
    Decompose(shard);

    return result;
}

/// Measure permutation state of a register
bitCapInt QRegister::MReg(bitLenInt start, bitLenInt length)
{
    bitLenInt end = start + length;
    bitCapInt result = 0;

    for (bitLenInt bit = start; bit < end; bit++) {
        result |= M(bit) << bit;
    }

    return result;
}

void QRegister::SetBit(bitLenInt qubit, bool value)
{
    QuantumBitShard &shard = shards[qubit];
    shard.unit->SetBit(shard.mapped, value);
    Decompose(shard);
}

/// Set register bits to given permutation
void QRegister::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    for (bitLenInt bit = start; bit < length; bit++) {
        QuantumBitShard &shard = shards[bit];
        shard.unit->SetBit(shard.mapped, value & (1 << bit));
        Decompose(shard);
    }
}

void QRegister::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    QuantumBitShard &shard1 = shards[qubit1];
    QuantumBitShard &shard2 = shards[qubit2];

    QuantumBitShard tmp;

    // Swap the bit mapping.
    tmp.mapped = shard1.mapped;
    shard1.mapped = shard2.mapped;
    shard2.mapped = tmp.mapped;

    // Swap the QUnit object.
    tmp.unit = shard1.unit;
    shard1.unit = shard2.unit;
    shard2.unit = tmp.unit;
}

void QRegister::Swap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt i = 0; i < length; i++) {
        Swap(qubit1 + i, qubit2 + i);
    }
}

void QRegister::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCall(inputBit1, inputBit2, outputBit, &QUnit::AND);
}

void QRegister::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCall(inputBit1, inputBit2, outputBit, &QUnit::OR);
}

void QRegister::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCall(inputBit1, inputBit2, outputBit, &QUnit::XOR);
}

void QRegister::CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputQBit)
{
    EntangleAndCall(inputBit1, inputBit2, [&](QUnit *unit, bitLenInt b1, bitLenInt b2) {
            unit->CLAND(b1, inputClassicalBit, b2);
        });
}

void QRegister::CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputQBit)
{
    EntangleAndCall(inputBit1, inputBit2, [&](QUnit *unit, bitLenInt b1, bitLenInt b2) {
            unit->CLOR(b1, inputClassicalBit, b2);
        });
}

void QRegister::CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputQBit)
{
    EntangleAndCall(inputBit1, inputBit2, [&](QUnit *unit, bitLenInt b1, bitLenInt b2) {
            unit->CLXOR(b1, inputClassicalBit, b2);
        });
}

void QRegister::CCNOT(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCall(inputBit1, inputBit2, outputBit, &QUnit::CCNOT);
}

void QRegister::AntiCCNOT(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCall(inputBit1, inputBit2, outputBit, &QUnit::AntiCCNOT);
}

void QRegister::H(bitLenInt qubit)
{
    shards[qubit]->unit->H(shards[qubit].mapped);
}

void QRegister::X(bitLenInt qubit)
{
    shards[qubit]->unit->X(shards[qubit].mapped);
}

void QRegister::Y(bitLenInt qubit)
{
    shards[qubit]->unit->Y(shards[qubit].mapped);
}

void QRegister::Z(bitLenInt qubit)
{
    shards[qubit]->unit->Z(shards[qubit].mapped);
}

void QRegister::X(bitLenInt start, bitLenInt length)
{
    for (bitLenInt i = 0; i < length; i++) {
        X(start + i);
    }
}

void QRegister::CY(bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, &QUnit::CY);
}

void QRegister::CZ(bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, &QUnit::CZ);
}

void QRegister::RT(double radians, bitLenInt qubit)
{
    EntangleAndCall(qubit, [&](QUnit *unit, bitLenInt q) {
            unit->RT(radians, q);
        });
}

void QRegister::RTDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit]->unit->RTDyad(numerator, denominator, shards[qubit].mapped);
}

void QRegister::RX(double radians, bitLenInt qubit)
{
    shards[qubit]->unit->RX(radians, shards[qubit].mapped);
}

void QRegister::RXDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit]->unit->RXDyad(numerator, denominator, shards[qubit].mapped);
}

void QRegister::RY(double radians, bitLenInt qubit)
{
    shards[qubit]->unit->RY(radians, shards[qubit].mapped);
}

void QRegister::RYDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit]->unit->RYDyad(numerator, denominator, shards[qubit].mapped);
}

// XXX XXX XXX Didn't make any further changes below here...

void QRegister::RZ(double radians, bitLenInt qubit)
{
    shards[qubit]->RZ(radians, qubitLookup[qubit].qb);
}

void QRegister::RZDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit]->RZDyad(numerator, denominator, qubitLookup[qubit].qb);
}

void QRegister::CRT(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, [&](QUnit *unit, bitLenInt c, bitLenInt t) {
            unit->CRT(radians, c, t);
        });
}

void QRegister::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, [&](QUnit *unit, bitLenInt c, bitLenInt t) {
            unit->CRTDyad(numerator, denominator, c, t);
        });
}

void QRegister::CRY(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, [&](QUnit *unit, bitLenInt c, bitLenInt t) {
            unit->CRY(radians, c, t);
        });
}

void QRegister::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, [&](QUnit *unit, bitLenInt c, bitLenInt t) {
            unit->CRYDyad(numerator, denominator, c, t);
        });
}

void QRegister::CRZ(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, [&](QUnit *unit, bitLenInt c, bitLenInt t) {
            unit->CRZ(radians, c, t);
        });
}

void QRegister::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    EntangleAndCall(control, target, [&](QUnit *unit, bitLenInt c, bitLenInt t) {
            unit->CRZDyad(numerator, denominator, c, t);
        });
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QRegister::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Reverse(start, end);
            Reverse(start, start + shift);
            Reverse(start + shift, end);
        }
    }
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QRegister::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Reverse(start + shift, end);
            Reverse(start, start + shift);
            Reverse(start, end);
        }
    }
}

void QRegister::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::INCSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::DEC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::DECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::DECSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::DECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::QFT(bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::PhaseFlip()
{
    for (shard : shards) {
        shard.unit->PhaseFlip();
    }
}

unsigned char QRegister::SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values)
{
    /* XXX TODO Map arbitrary list. */
}

unsigned char QRegister::AdcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    /* XXX TODO Map arbitrary list. */
}

unsigned char QRegister::SbcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    /* XXX TODO Map arbitrary list. */
}

void QRegister::GetOrderedBitList(bitLenInt start, bitLenInt length, std::vector<QbListEntry>& qbList)
{
    /* XXX TODO Map arbitrary list. */
}

} // namespace Qrack
