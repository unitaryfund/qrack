//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a fill-in over the std::complex<__fp16> type, for Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <cmath>

#include "common/complex2.hpp"

float real(const Qrack::Complex2& c) { return c.real; }

float imag(const Qrack::Complex2& c) { return c.imag; }

float abs(const Qrack::Complex2& c) { return sqrt(c.real * c.real + c.imag * c.imag); }

float arg(const Qrack::Complex2& c) { return atan2(c.real, c.imag); }

float norm(const Qrack::Complex2& c) { return c.real * c.real + c.imag * c.imag; }

Qrack::Complex2 sqrt(const Qrack::Complex2& c) { return sqrt(std::complex<float>(c.real, c.imag)); }

Qrack::Complex2 exp(const Qrack::Complex2& c) { return exp(std::complex<float>(c.real, c.imag)); }

Qrack::Complex2 pow(const Qrack::Complex2& b, const Qrack::Complex2& p)
{
    return pow(std::complex<float>(b.real, b.imag), std::complex<float>(p.real, p.imag));
}

Qrack::Complex2 conj(const Qrack::Complex2& c) { return Qrack::Complex2(c.real, -c.imag); }
