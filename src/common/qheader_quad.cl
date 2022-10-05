//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp128 : enable

#define cmplx quad2
#define cmplx2 quad4
#define cmplx4 quad8
#define real1 quad
#define real1_f double
#define ZERO_R1 ((real1)0.0)
#define ONE_R1 ((real1)1.0)
#define SineShift ((real1)M_PI_2_F)
#define PI_R1 ((real1)M_PI_F)
#define REAL1_EPSILON ((real1)2e-129)
