//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <algorithm>

#include "qinterface.hpp"

namespace Qrack {

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride)
{
    while ((first < last) && (first < (last - stride))) {
        last -= stride;
        std::iter_swap(first, last);
        first += stride;
    }
}

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride)
{
    reverse(first, middle, stride);
    reverse(middle, last, stride);
    reverse(first, last, stride);
}

template void rotate<complex*>(complex* first, complex* middle, complex* last, bitCapInt stride);

} // namespace Qrack
