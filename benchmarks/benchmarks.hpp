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

#include <iomanip>
#include <sstream>

#include "qinterface.hpp"

#include "catch.hpp"

/*
 * A fixture to create a unique QInterface test, of the appropriate type, for
 * each executing test case.
 */
class QInterfaceTestFixture {
protected:
    Qrack::QInterfacePtr qftReg;
    Qrack::QInterfaceEngine engineType;
    Qrack::QInterfaceEngine subEngineType;
    std::shared_ptr<std::default_random_engine> rng;
public:
    QInterfaceTestFixture();
};
