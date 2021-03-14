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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define cmplx double2
#define cmplx2 double4
#define cmplx4 double8
#define real1 double
#define real1_f double
#define ZERO_R1 0.0
#define ONE_R1 1.0
#define SineShift M_PI_2
#define PI_R1 M_PI
#define REAL1_EPSILON 2e-65
