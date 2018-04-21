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

#pragma once

#include <functional>

/* Needed for bitCapInt typedefs. */
#include "qinterface.hpp"

namespace Qrack {

class ParallelFor 
{
private:
    int32_t numCores;

public:
    ParallelFor() : numCores(1) { }
    virtual ~ParallelFor() { }

    void SetConcurrencyLevel(int32_t num) { numCores = num; }
    int32_t GetConcurrencyLevel() { return numCores; }
    /*
     * Parallelization routines for spreading work across multiple cores.
     */

    /** Called once per value between begin and end. */
    typedef std::function<void(const bitCapInt, const int cpu)> ParallelFunc;
    typedef std::function<bitCapInt(const bitCapInt, const int cpu)> IncrementFunc;

    /**
     * Iterate through the permutations a maximum of end-begin times, allowing
     * the caller to control the incrementation offset through 'inc'.
     */
    void par_for_inc(const bitCapInt begin, const bitCapInt end, IncrementFunc, ParallelFunc fn);

    /** Call fn once for every numerical value between begin and end. */
    void par_for(const bitCapInt begin, const bitCapInt end, ParallelFunc fn);

    /**
     * Skip over the skipPower bits.
     *
     * For example, if skipPower is 2, it will count:
     *   0000, 0001, 0100, 0101, 1000, 1001, 1100, 1101.
     *     ^     ^     ^     ^     ^     ^     ^     ^ - The second bit is
     *                                                   untouched.
     */
    void par_for_skip(const bitCapInt begin, const bitCapInt end, const bitCapInt skipPower,
        const bitLenInt skipBitCount, ParallelFunc fn);

    /** Skip over the bits listed in maskArray in the same fashion as par_for_skip. */
    void par_for_mask(
        const bitCapInt, const bitCapInt, const bitCapInt* maskArray, const bitLenInt maskLen, ParallelFunc fn);

    /** Calculate the normal for the array. */
    double par_norm(const bitCapInt maxQPower, const Complex16* stateArray);
};

}
