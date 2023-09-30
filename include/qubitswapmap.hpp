//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QubitSwapMap enables constant complexity SWAP gates, via qubit label swap.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/qrack_types.hpp"

namespace Qrack {

class QubitSwapMap {
protected:
    std::vector<bitLenInt> swapMap;

public:
    QubitSwapMap()
    {
        // Intentionally left blank
    }

    QubitSwapMap(bitLenInt qubitCount)
        : swapMap(qubitCount)
    {
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            swapMap[i] = i;
        }
    }

    typedef std::vector<bitLenInt>::iterator iterator;

    bitLenInt& operator[](const bitLenInt& i) { return swapMap[i]; }

    bitLenInt size() { return swapMap.size(); }

    void swap(bitLenInt qubit1, bitLenInt qubit2) { std::swap(swapMap[qubit1], swapMap[qubit2]); }

    bitCapInt map(bitCapInt perm)
    {
        bitCapInt toRet = 0U;
        for (bitLenInt i = 0U; i < size(); ++i) {
            if ((perm >> i) & 1U) {
                toRet |= (ONE_BCI << swapMap[i]);
            }
        }
        return toRet;
    }

    bitCapInt inverseMap(bitCapInt perm)
    {
        bitCapInt toRet = 0U;
        for (bitLenInt i = 0U; i < size(); ++i) {
            if ((perm >> swapMap[i]) & 1U) {
                toRet |= (ONE_BCI << i);
            }
        }
        return toRet;
    }
};

} // namespace Qrack
