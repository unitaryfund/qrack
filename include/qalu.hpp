//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/qrack_functions.hpp"

namespace Qrack {

class QAlu;
typedef std::shared_ptr<QAlu> QAluPtr;

class QAlu {
public:
    virtual bool M(bitLenInt qubitIndex) = 0;
    virtual void X(bitLenInt qubitIndex) = 0;

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    /** This is an expedient for an adaptive Grover's search for a function's global minimum. */
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length) = 0;
    /** The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation. */
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) = 0;

    /** Add integer (without sign) */
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;
    /** Add integer (without sign) */
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) = 0;
    /** Add integer (without sign, with controls) */
    virtual void CINC(bitCapInt toAdd, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls) = 0;
    /** Subtract integer (without sign, with controls) */
    virtual void CDEC(bitCapInt toSub, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls);
    /** Add integer (without sign, with carry) */
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    /** Subtract classical integer (without sign, with carry) */
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    /** Common driver method behind INCC and DECC (without sign, with carry) */
    virtual void INCDECC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    /** Add a classical integer to the register, with sign and without carry. */
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) = 0;
    /** Add a classical integer to the register, with sign and without carry. */
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) = 0;
    /** Add a classical integer to the register, with sign and with carry. */
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    /** Add a classical integer to the register, with sign and with (phase-based) carry. */
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    /** Subtract a classical integer from the register, with sign and with carry. */
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    /** Common driver method behind INCSC and DECSC (without overflow flag) */
    virtual void INCDECSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    /** Common driver method behind INCSC and DECSC (with overflow flag) */
    virtual void INCDECSC(
        bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;
    /** Multiply by integer */
    virtual void MUL(bitCapInt toMul, bitLenInt start, bitLenInt carryStart, bitLenInt length) = 0;
    /** Divide by integer */
    virtual void DIV(bitCapInt toDiv, bitLenInt start, bitLenInt carryStart, bitLenInt length) = 0;
    /** Multiplication modulo N by integer, (out of place) */
    virtual void MULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    /** Inverse of multiplication modulo N by integer, (out of place) */
    virtual void IMULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    /** Raise a classical base to a quantum power, modulo N, (out of place) */
    virtual void POWModNOut(
        bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    /** Controlled multiplication by integer */
    virtual void CMUL(bitCapInt toMul, bitLenInt start, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls) = 0;
    /** Controlled division by power of integer */
    virtual void CDIV(bitCapInt toDiv, bitLenInt start, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls) = 0;
    /** Controlled multiplication modulo N by integer, (out of place) */
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls) = 0;
    /** Inverse of controlled multiplication modulo N by integer, (out of place) */
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls) = 0;
    /** Controlled, raise a classical base to a quantum power, modulo N, (out of place) */
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls) = 0;

#if ENABLE_BCD
    /** Add classical BCD integer (without sign) */
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;
    /** Subtract classical BCD integer (without sign) */
    virtual void DECBCD(bitCapInt toSub, bitLenInt start, bitLenInt length);
    /** Common driver method behind INCSC and DECSC (without overflow flag) */
    virtual void INCDECBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
#endif

    /**
     * Set 8 bit register bits by a superposed index-offset-based read from
     * classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits are first cleared,
     * then the separable |input, 00000000> permutation state is mapped to
     * |input, values[input]>, with "values[input]" placed in the "outputStart"
     * register. FOR BEST EFFICIENCY, the "values" array should be allocated aligned to a 64-byte boundary. (See the
     * unit tests suite code for an example of how to align the allocation.)
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM.
     *
     * The physical motivation for this addressing mode can be explained as
     * follows: say that we have a superconducting quantum interface device
     * (SQUID) based chip. SQUIDs have already been demonstrated passing
     * coherently superposed electrical currents. In a sufficiently
     * quantum-mechanically isolated qubit chip with a classical cache, with
     * both classical RAM and registers likely cryogenically isolated from the
     * environment, SQUIDs could (hopefully) pass coherently superposed
     * electrical currents into the classical RAM cache to load values into a
     * qubit register. The state loaded would be a superposition of the values
     * of all RAM to which coherently superposed electrical currents were
     * passed.
     *
     * In qubit system similar to the MOS 6502, say we have qubit-based
     * "accumulator" and "X index" registers, and say that we start with a
     * superposed X index register. In (classical) X addressing mode, the X
     * index register value acts an offset into RAM from a specified starting
     * address. The X addressing mode of a LoaD Accumulator (LDA) instruction,
     * by the physical mechanism described above, should load the accumulator
     * in quantum parallel with the values of every different address of RAM
     * pointed to in superposition by the X index register. The superposed
     * values in the accumulator are entangled with those in the X index
     * register, by way of whatever values the classical RAM pointed to by X
     * held at the time of the load. (If the RAM at index "36" held an unsigned
     * char value of "27," then the value "36" in the X index register becomes
     * entangled with the value "27" in the accumulator, and so on in quantum
     * parallel for all superposed values of the X index register, at once.) If
     * the X index register or accumulator are then measured, the two registers
     * will both always collapse into a random but valid key-value pair of X
     * index offset and value at that classical RAM address.
     *
     * Note that a "superposed store operation in classical RAM" is not
     * possible by analagous reasoning. Classical RAM would become entangled
     * with both the accumulator and the X register. When the state of the
     * registers was collapsed, we would find that only one "store" operation
     * to a single memory address had actually been carried out, consistent
     * with the address offset in the collapsed X register and the byte value
     * in the collapsed accumulator. It would not be possible by this model to
     * write in quantum parallel to more than one address of classical memory
     * at a time.
     */
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true) = 0;
    /**
     * Add to entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits would usually
     * already be entangled with the "inputStart" bits via a IndexedLDA()
     * operation. With the "inputStart" bits being a "key" and the
     * "outputStart" bits being a value, the permutation state |key, value> is
     * mapped to |key, value + values[key]>. This is similar to classical
     * parallel addition of two arrays.  However, when either of the registers
     * are measured, both registers will collapse into one random VALID
     * key-value pair, with any addition or subtraction done to the "value."
     * See IndexedLDA() for context.
     *
     * FOR BEST EFFICIENCY, the "values" array should be allocated aligned to a 64-byte boundary. (See the unit tests
     * suite code for an example of how to align the allocation.)
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "IndexedADC" and "IndexedSBC" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * IndexedLDA().
     */
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values) = 0;
    /**
     * Subtract from an entangled 8 bit register state with a superposed
     * index-offset-based read from classical memory
     *
     * "inputStart" is the start index of 8 qubits that act as an index into
     * the 256 byte "values" array. The "outputStart" bits would usually
     * already be entangled with the "inputStart" bits via a IndexedLDA()
     * operation.  With the "inputStart" bits being a "key" and the
     * "outputStart" bits being a value, the permutation state |key, value> is
     * mapped to |key, value - values[key]>. This is similar to classical
     * parallel addition of two arrays.  However, when either of the registers
     * are measured, both registers will collapse into one random VALID
     * key-value pair, with any addition or subtraction done to the "value."
     * See QInterface::IndexedLDA for context.
     *
     * FOR BEST EFFICIENCY, the "values" array should be allocated aligned to a 64-byte boundary. (See the unit tests
     * suite code for an example of how to align the allocation.)
     *
     * While a QInterface represents an interacting set of qubit-based
     * registers, or a virtual quantum chip, the registers need to interact in
     * some way with (classical or quantum) RAM. IndexedLDA is a RAM access
     * method similar to the X addressing mode of the MOS 6502 chip, if the X
     * register can be in a state of coherent superposition when it loads from
     * RAM. "IndexedADC" and "IndexedSBC" perform add and subtract
     * (with carry) operations on a state usually initially prepared with
     * IndexedLDA().
     */
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values) = 0;
    /**
     * Transform a length of qubit register via lookup through a hash table. The hash table must be a one-to-one
     * function, otherwise the behavior of this method is undefined. The value array definition convention is the same
     * as IndexedLDA(). Essentially, this is an IndexedLDA() operation that replaces the index register with the value
     * register, but the lookup table must therefore be one-to-one, for this operation to be unitary, as required.
     */
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values) = 0;

#if ENABLE_BCD

    /** Add classical BCD integer (without sign, with carry) */
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** Subtract BCD integer (without sign, with carry) */
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#endif

    /** @} */
};
} // namespace Qrack
