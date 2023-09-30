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

#include <string>

namespace Qrack {

enum SPECIAL_2X2 { NONE = 0, PAULIX, PAULIZ, INVERT, PHASE };

class bad_alloc : public std::bad_alloc {
private:
    std::string m;

public:
    bad_alloc(std::string message)
        : m(message)
    {
        // Intentionally left blank.
    }

    const char* what() const noexcept { return m.c_str(); }
};
} // namespace Qrack
