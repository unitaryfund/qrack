#include <cmath>
#include <complex>

namespace Qrack {

struct Complex2 {
    __fp16 real;
    __fp16 imag;

    Complex2() {}

    Complex2(float r, float i)
        : real(r)
        , imag(i)
    {
    }

    Complex2(std::complex<float> o)
        : real(std::real(o))
        , imag(std::imag(o))
    {
    }

    Complex2(float r)
        : real(r)
        , imag(0)
    {
    }

    bool operator==(const Complex2& rhs) const { return (real == rhs.real) && (imag == rhs.imag); }

    bool operator!=(const Complex2& rhs) const { return (real != rhs.real) || (imag != rhs.imag); }

    Complex2 operator-() const { return Complex2(-real, -imag); }

    inline Complex2 operator+(const Complex2& rhs) const { return Complex2(real + rhs.real, imag + rhs.imag); }

    inline Complex2 operator-(const Complex2& rhs) const { return Complex2(real - rhs.real, imag - rhs.imag); }

    inline Complex2 operator*(const Complex2& rhs) const
    {
        return Complex2(real * rhs.real - imag * rhs.imag, real * rhs.imag + imag * rhs.real);
    }

    inline Complex2 operator*(const float& rhs) const { return Complex2(real * rhs, imag * rhs); }

    inline Complex2 operator*=(const float& rhs) { return Complex2(real *= rhs, imag *= rhs); }

    inline Complex2 operator*=(const Complex2& rhs)
    {
        Complex2 temp(real * rhs.real - imag * rhs.imag, real * rhs.imag + imag * rhs.real);
        real = temp.real;
        imag = temp.imag;
        return temp;
    }

    inline Complex2 operator/(const Complex2& rhs) const
    {
        return (Complex2(real, imag) * rhs) / (rhs.real * rhs.real + rhs.imag * rhs.imag);
    }

    inline Complex2 operator/(const float& rhs) const { return Complex2(real / rhs, imag / rhs); }

    inline Complex2 operator/=(const float& rhs) { return Complex2(real /= rhs, imag /= rhs); }

    inline Complex2 operator/=(const Complex2& rhs)
    {
        Complex2 temp = (Complex2(real, imag) * rhs) / (rhs.real * rhs.real + rhs.imag * rhs.imag);
        real = temp.real;
        imag = temp.imag;
        return temp;
    }
};

float real(const Complex2& c) { return c.real; }

float imag(const Complex2& c) { return c.imag; }

float abs(const Complex2& c) { return sqrt(c.real * c.real + c.imag * c.imag); }

float arg(const Complex2& c) { return atan2(c.real, c.imag); }

} // namespace Qrack

inline float norm(const Qrack::Complex2& c) { return c.real * c.real + c.imag * c.imag; }

inline Qrack::Complex2 sqrt(const Qrack::Complex2& c) { return sqrt(std::complex<float>(c.real, c.imag)); }

inline Qrack::Complex2 exp(const Qrack::Complex2& c) { return exp(std::complex<float>(c.real, c.imag)); }

inline Qrack::Complex2 pow(const Qrack::Complex2& b, const Qrack::Complex2& p)
{
    return pow(std::complex<float>(b.real, b.imag), std::complex<float>(p.real, p.imag));
}

namespace Qrack {

inline Complex2 operator*(const float& lhs, const Complex2& rhs) { return Complex2(lhs * rhs.real, lhs * rhs.imag); }

inline Complex2 operator/(const float& lhs, const Complex2& rhs) { return (lhs * rhs) / norm(rhs); }

} // namespace Qrack
