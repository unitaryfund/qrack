//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>
#include <set>
#include <vector>

/* Needed for bitCapInt typedefs. */
#include "qrack_types.hpp"

namespace Qrack {

class ParallelFor {
private:
    int32_t numCores;

public:
    ParallelFor()
        : numCores(1)
    {
    }
    virtual ~ParallelFor() {}

    void SetConcurrencyLevel(int32_t num) { numCores = num; }
    int32_t GetConcurrencyLevel() { return numCores; }
    /*
     * Parallelization routines for spreading work across multiple cores.
     */

    /**
     * Iterate through the permutations a maximum of end-begin times, allowing
     * the caller to control the incrementation offset through 'inc'.
     */
    void par_for_inc(const bitCapInt begin, const bitCapInt itemCount, IncrementFunc, ParallelFunc fn);

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

    /** Iterate over a sparse state vector. */
    void par_for_set(const std::set<bitCapInt>& sparseSet, ParallelFunc fn);

    /** Iterate over a sparse state vector. */
    void par_for_set(const std::vector<bitCapInt>& sparseSet, ParallelFunc fn);

    /** Iterate over the power set of 2 sparse state vectors. */
    void par_for_sparse_compose(const std::vector<bitCapInt>& lowSet, const std::vector<bitCapInt>& highSet,
        const bitLenInt& highStart, ParallelFunc fn);

    /** Calculate the normal for the array, (with flooring). */
    real1_f par_norm(const bitCapInt maxQPower, const StateVectorPtr stateArray, real1_f norm_thresh = ZERO_R1);

    /** Calculate the normal for the array, (without flooring. */
    real1_f par_norm_exact(const bitCapInt maxQPower, const StateVectorPtr stateArray);
};

} // namespace Qrack
