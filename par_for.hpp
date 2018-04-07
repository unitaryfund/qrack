//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include <atomic>
#include <future>
#include <thread>

#include "qregister.hpp"

namespace Qrack {

typedef std::function<void(const bitCapInt)> ParallelFunc;

void par_for(const bitCapInt begin, const bitCapInt end, ParallelFunc fn);

void par_for_skip(const bitCapInt begin, const bitCapInt end, const bitCapInt skipPower, const bitLenInt skipBitCount,
    ParallelFunc fn);

void par_for_mask(const bitCapInt, const bitCapInt, const bitCapInt* maskArray, const bitLenInt bitCount, ParallelFunc);

double par_norm(const bitCapInt maxQPower, const Complex16* stateArray);

} // namespace Qrack
