// From: https://github.com/eteran/cpp-utilities/blob/master/fixed/include/cpp-utilities/fixed.h
// See also: http://stackoverflow.com/questions/79677/whats-the-best-way-to-do-fixed-point-math
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Evan Teran
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "common/fixed.hpp"

#if __cplusplus >= 201402L
#define CONSTEXPR14 constexpr
#else
#define CONSTEXPR14
#endif

#include <cstddef> // for size_t
#include <cstdint>
#include <exception>
#include <ostream>
#include <type_traits>

namespace numeric {

// if we have the same fractional portion, but differing integer portions, we trivially upgrade the smaller type
template <size_t I1, size_t I2, size_t F>
CONSTEXPR14 typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type operator+(
    fixed<I1, F> lhs, fixed<I2, F> rhs)
{

    using T = typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type;

    const T l = T::from_base(lhs.to_raw());
    const T r = T::from_base(rhs.to_raw());
    return l + r;
}

template <size_t I1, size_t I2, size_t F>
CONSTEXPR14 typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type operator-(
    fixed<I1, F> lhs, fixed<I2, F> rhs)
{

    using T = typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type;

    const T l = T::from_base(lhs.to_raw());
    const T r = T::from_base(rhs.to_raw());
    return l - r;
}

template <size_t I1, size_t I2, size_t F>
CONSTEXPR14 typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type operator*(
    fixed<I1, F> lhs, fixed<I2, F> rhs)
{

    using T = typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type;

    const T l = T::from_base(lhs.to_raw());
    const T r = T::from_base(rhs.to_raw());
    return l * r;
}

template <size_t I1, size_t I2, size_t F>
CONSTEXPR14 typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type operator/(
    fixed<I1, F> lhs, fixed<I2, F> rhs)
{

    using T = typename std::conditional<I1 >= I2, fixed<I1, F>, fixed<I2, F>>::type;

    const T l = T::from_base(lhs.to_raw());
    const T r = T::from_base(rhs.to_raw());
    return l / r;
}

template <size_t I, size_t F> std::ostream& operator<<(std::ostream& os, fixed<I, F> f)
{
    os << f.to_double();
    return os;
}

// basic math operators
template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator+(fixed<I, F> lhs, Number rhs)
{
    lhs += fixed<I, F>(rhs);
    return lhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator-(fixed<I, F> lhs, Number rhs)
{
    lhs -= fixed<I, F>(rhs);
    return lhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator*(fixed<I, F> lhs, Number rhs)
{
    lhs *= fixed<I, F>(rhs);
    return lhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator/(fixed<I, F> lhs, Number rhs)
{
    lhs /= fixed<I, F>(rhs);
    return lhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator+(Number lhs, fixed<I, F> rhs)
{
    fixed<I, F> tmp(lhs);
    tmp += rhs;
    return tmp;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator-(Number lhs, fixed<I, F> rhs)
{
    fixed<I, F> tmp(lhs);
    tmp -= rhs;
    return tmp;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator*(Number lhs, fixed<I, F> rhs)
{
    fixed<I, F> tmp(lhs);
    tmp *= rhs;
    return tmp;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
CONSTEXPR14 fixed<I, F> operator/(Number lhs, fixed<I, F> rhs)
{
    fixed<I, F> tmp(lhs);
    tmp /= rhs;
    return tmp;
}

// shift operators
template <size_t I, size_t F, class Integer, class = typename std::enable_if<std::is_integral<Integer>::value>::type>
CONSTEXPR14 fixed<I, F> operator<<(fixed<I, F> lhs, Integer rhs)
{
    lhs <<= rhs;
    return lhs;
}

template <size_t I, size_t F, class Integer, class = typename std::enable_if<std::is_integral<Integer>::value>::type>
CONSTEXPR14 fixed<I, F> operator>>(fixed<I, F> lhs, Integer rhs)
{
    lhs >>= rhs;
    return lhs;
}

// comparison operators
template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator>(fixed<I, F> lhs, Number rhs)
{
    return lhs > fixed<I, F>(rhs);
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator<(fixed<I, F> lhs, Number rhs)
{
    return lhs < fixed<I, F>(rhs);
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator>=(fixed<I, F> lhs, Number rhs)
{
    return lhs >= fixed<I, F>(rhs);
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator<=(fixed<I, F> lhs, Number rhs)
{
    return lhs <= fixed<I, F>(rhs);
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator==(fixed<I, F> lhs, Number rhs)
{
    return lhs == fixed<I, F>(rhs);
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator!=(fixed<I, F> lhs, Number rhs)
{
    return lhs != fixed<I, F>(rhs);
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator>(Number lhs, fixed<I, F> rhs)
{
    return fixed<I, F>(lhs) > rhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator<(Number lhs, fixed<I, F> rhs)
{
    return fixed<I, F>(lhs) < rhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator>=(Number lhs, fixed<I, F> rhs)
{
    return fixed<I, F>(lhs) >= rhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator<=(Number lhs, fixed<I, F> rhs)
{
    return fixed<I, F>(lhs) <= rhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator==(Number lhs, fixed<I, F> rhs)
{
    return fixed<I, F>(lhs) == rhs;
}

template <size_t I, size_t F, class Number, class = typename std::enable_if<std::is_arithmetic<Number>::value>::type>
constexpr bool operator!=(Number lhs, fixed<I, F> rhs)
{
    return fixed<I, F>(lhs) != rhs;
}

} // namespace numeric
