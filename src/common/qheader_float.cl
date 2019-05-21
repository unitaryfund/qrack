//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
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
#define ZERO_R1 0.0f
#define SineShift M_PI_2_F
#define PI_R1 M_PI_F
#define min_norm 0
#define bitCapInt ulong
#define bitCapInt2 ulong2
#define bitCapInt4 ulong4
#define bitLenInt unsigned char
