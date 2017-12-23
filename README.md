Copyright (c) Daniel Strano 2017. All rights reserved. (See "par_for.hpp" for additional information.)
Licensed under the GNU General Public License V3.
See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html for details.

This is a header-only, quick-and-dirty, multithreaded, universal quantum register simulation, allowing (nonphysical) register cloning and direct measurement of probability and phase, to leverage what advantages classical emulation of qubits can have.

To use:
1)#include "qrack.hpp"
2)Link against math and pthreads. (See build.sh for example.)

Instantiate a Qrack::Register, specifying the desired number of qubits. (Optionally, also specify the initial bit state in the constructor.)

For more information, compile the doxygen.config in the root folder, and then check the "doc" folder.

EXAMPLE.CPP:

This is a simple example of quantum mechanics simulation in quantum computational logic. It is essentially a unidirectional binary quantum random walk algorithm, from a positive starting point, heading toward zero.

We assume a fixed length time step. During each time step, we step through an equal superposition of either standing still or taking one fixed length step from our current position toward our fixed destination.

This is equivalent to a physical body having a 50% chance of emitting a fixed unit of energy per a fixed unit of time, in a pure quantum state. Hence, it might be considered a simple quantum mechanics simulation.
