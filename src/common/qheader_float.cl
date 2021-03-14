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

#define cmplx float2
#define cmplx2 float4
#define cmplx4 float8
#define real1 float
#define real1_f float
#define ZERO_R1 0.0f
#define ONE_R1 1.0f
#define SineShift M_PI_2_F
#define PI_R1 M_PI_F
#define REAL1_EPSILON 2e-33f
