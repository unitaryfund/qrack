#pragma once
#include "common/qrack_types.hpp"

namespace Qrack {

struct MpsShard;
typedef std::shared_ptr<MpsShard> MpsShardPtr;

struct MpsShard {
    complex gate[4];

    MpsShard()
    {
        gate[0] = ONE_CMPLX;
        gate[1] = ZERO_CMPLX;
        gate[2] = ZERO_CMPLX;
        gate[3] = ONE_CMPLX;
    }

    MpsShard(complex* g) { std::copy(g, g + 4, gate); }

    void Compose(const complex* g)
    {
        complex o[4];
        std::copy(gate, gate + 4, o);
        mul2x2((complex*)g, o, gate);
    }

    bool IsPhase() { return (norm(gate[1]) <= FP_NORM_EPSILON) && (norm(gate[2]) <= FP_NORM_EPSILON); }

    bool IsInvert() { return (norm(gate[0]) <= FP_NORM_EPSILON) && (norm(gate[3]) <= FP_NORM_EPSILON); }

    bool IsIdentity() { return IsPhase() && (norm(gate[0] - gate[3]) <= FP_NORM_EPSILON); }

    bool IsX(bool randGlobalPhase = true)
    {
        return IsInvert() && (norm(gate[1] - gate[2]) <= FP_NORM_EPSILON) &&
            (randGlobalPhase || (norm(ONE_CMPLX - gate[1]) <= FP_NORM_EPSILON));
    }

    bool IsY(bool randGlobalPhase = true)
    {
        return IsInvert() && (norm(gate[1] + gate[2]) <= FP_NORM_EPSILON) &&
            (randGlobalPhase || (norm(ONE_CMPLX + gate[1]) <= FP_NORM_EPSILON));
    }

    bool IsZ(bool randGlobalPhase = true)
    {
        return IsPhase() && (norm(gate[0] + gate[3]) <= FP_NORM_EPSILON) &&
            (randGlobalPhase || (norm(ONE_CMPLX - gate[0]) <= FP_NORM_EPSILON));
    }
};

} // namespace Qrack
