#include <stdint.h>
#include <vector>
#include <memory>
#include <emmintrin.h>
#include "VapourSynth.h"
#include "VSHelper.h"

struct YUVPixel {
    uint8_t Y;
    uint8_t U;
    uint8_t V;

    uint32_t vector_y;
    uint32_t vector_u;
    uint32_t vector_v;
};

struct TCMData {
    VSNodeRef * node;
    VSVideoInfo vi;

    std::vector<YUVPixel> colors;
    int tolerance;
    bool bt601;
    bool grayscale;
    int prefer_lut_thresh;
    int subsamplingY;
    int subsamplingX;
    uint32_t vector_tolerance;
    uint32_t vector_half_tolerance;

    uint8_t lut_y[256];
    uint8_t lut_u[256];
    uint8_t lut_v[256];

    void(*lutFunction)(uint8_t *pDstY, const uint8_t *pSrcY, const uint8_t *pSrcV, const uint8_t *pSrcU, int dstPitchY, int srcPitchY, int srcPitchUV, int width, int height, uint8_t *lutY, uint8_t *lutU, uint8_t *lutV);
    void(*sse2Function)(uint8_t *pDstY, const uint8_t *pSrcY, const uint8_t *pSrcV, const uint8_t *pSrcU, int dstPitchY, int srcPitchY, int srcPitchUV, int width, int height, const std::vector<YUVPixel>& colors, int vectorTolerance, int halfVectorTolerance);
};

inline int depfree_round(float d) {
    return static_cast<int>(d + 0.5f);
}

template<int subsamplingX, int subsamplingY>
void processLut(uint8_t *pDstY, const uint8_t *pSrcY, const uint8_t *pSrcV, const uint8_t *pSrcU, int dstPitchY, int srcPitchY, int srcPitchUV, int width, int height, uint8_t *lutY, uint8_t *lutU, uint8_t *lutV) {
    for(int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pDstY[x] = lutY[pSrcY[x]] & lutU[pSrcU[x/subsamplingX]] & lutV[pSrcV[x/subsamplingX]];
        }
        pSrcY += srcPitchY;
        if (y % subsamplingY == (subsamplingY-1)) {
            pSrcU += srcPitchUV;
            pSrcV += srcPitchUV;
        }
        pDstY += dstPitchY;
    }
}

template<int subsamplingX, int subsamplingY>
void processSse2(uint8_t *pDstY, const uint8_t *pSrcY, const uint8_t *pSrcV, const uint8_t *pSrcU, int dstPitchY, int srcPitchY, int srcPitchUV, int width, int height, const std::vector<YUVPixel>& colors, int vectorTolerance, int halfVectorTolerance) {
    for(int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x+=16) {
            auto result_y = _mm_setzero_si128();
            auto result_u = _mm_setzero_si128();
            auto result_v = _mm_setzero_si128();

            auto srcY_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pSrcY+x));
            __m128i srcU_v, srcV_v;
            if (subsamplingX == 2) {
                srcU_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrcU+x/subsamplingX));
                srcU_v = _mm_unpacklo_epi8(srcU_v, srcU_v);
                srcV_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrcV+x/subsamplingX));
                srcV_v = _mm_unpacklo_epi8(srcV_v, srcV_v);
            } else {
                srcU_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pSrcU+x));
                srcV_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pSrcV+x));
            }

            for(auto &color: colors) {
                auto colorVector_y = _mm_set1_epi32(color.vector_y);
                auto colorVector_u = _mm_set1_epi32(color.vector_u);
                auto colorVector_v = _mm_set1_epi32(color.vector_v);
                /* absolute difference */
                auto maximum_y = _mm_max_epu8(srcY_v, colorVector_y);
                auto maximum_u = _mm_max_epu8(srcU_v, colorVector_u);
                auto maximum_v = _mm_max_epu8(srcV_v, colorVector_v);

                auto minimum_y = _mm_min_epu8(srcY_v, colorVector_y);
                auto minimum_u = _mm_min_epu8(srcU_v, colorVector_u);
                auto minimum_v = _mm_min_epu8(srcV_v, colorVector_v);

                auto diff_y = _mm_subs_epu8(maximum_y, minimum_y);
                auto diff_u = _mm_subs_epu8(maximum_u, minimum_u);
                auto diff_v = _mm_subs_epu8(maximum_v, minimum_v);
                /* comparing to tolerance */
                auto diff_tolerance_min_y = _mm_max_epu8(diff_y, _mm_set1_epi32(vectorTolerance));
                auto diff_tolerance_min_u = _mm_max_epu8(diff_u, _mm_set1_epi32(halfVectorTolerance));
                auto diff_tolerance_min_v = _mm_max_epu8(diff_v, _mm_set1_epi32(halfVectorTolerance));

                auto passed_y = _mm_cmpeq_epi8(diff_y, diff_tolerance_min_y);
                auto passed_u = _mm_cmpeq_epi8(diff_u, diff_tolerance_min_u);
                auto passed_v = _mm_cmpeq_epi8(diff_v, diff_tolerance_min_v);
                /* inverting to get "lower" instead of "lower or equal" */
                passed_y = _mm_andnot_si128(passed_y, _mm_set1_epi32(0xFFFFFFFF));
                passed_u = _mm_andnot_si128(passed_u, _mm_set1_epi32(0xFFFFFFFF));
                passed_v = _mm_andnot_si128(passed_v, _mm_set1_epi32(0xFFFFFFFF));

                result_y = _mm_or_si128(result_y, passed_y);
                result_u = _mm_or_si128(result_u, passed_u);
                result_v = _mm_or_si128(result_v, passed_v);
            }
            result_y = _mm_and_si128(result_y, result_u);
            result_y = _mm_and_si128(result_y, result_v);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDstY+x), result_y);
        }
        pSrcY += srcPitchY;
        if (y % subsamplingY == (subsamplingY-1)) {
            pSrcU += srcPitchUV;
            pSrcV += srcPitchUV;
        }
        pDstY += dstPitchY;
    }
}

int stringToInt(const std::string &str) {
    if (str[0] == '$') {
        auto substr = str.substr(1, str.length());
        return strtol(substr.c_str(), 0, 16);
    }
    return strtol(str.c_str(), 0, 10);
}

template<typename pixel_t>
void process(const VSFrameRef * src, VSFrameRef * dst, int bits, pixel_t *lut_y, pixel_t *lut_u, pixel_t *lut_v, const TCMData * d, const VSAPI * vsapi) {
    const pixel_t *srcY_ptr = reinterpret_cast<const pixel_t *>(vsapi->getReadPtr(src, 0));
    const pixel_t *srcU_ptr = reinterpret_cast<const pixel_t *>(vsapi->getReadPtr(src, 1));
    const pixel_t *srcV_ptr = reinterpret_cast<const pixel_t *>(vsapi->getReadPtr(src, 2));
    pixel_t *VS_RESTRICT dstY_ptr = reinterpret_cast<pixel_t *>(vsapi->getWritePtr(dst, 0));
    const int srcStrideY = vsapi->getStride(src, 0) / sizeof(pixel_t);
    const int srcStrideUV = vsapi->getStride(src, 1) / sizeof(pixel_t);
    const int dstStride = vsapi->getStride(dst, 0) / sizeof(pixel_t);

    if (d->colors.size() > d->prefer_lut_thresh) {
        d->lutFunction(dstY_ptr, srcY_ptr, srcV_ptr, srcU_ptr, dstStride, srcStrideY, srcStrideUV, d->vi.width, d->vi.height, lut_y, lut_u, lut_v);
        return;
    }
    int border = d->vi.width % 16;

    d->sse2Function(dstY_ptr, srcY_ptr, srcV_ptr, srcU_ptr, dstStride, srcStrideY, srcStrideUV, d->vi.width - border, d->vi.height, d->colors, d->vector_tolerance, d->vector_half_tolerance);
    if (border != 0) {
        d->lutFunction(dstY_ptr + d->vi.width - border,
            srcY_ptr + (d->vi.width - border) / d->subsamplingX,
            srcV_ptr + (d->vi.width - border) / d->subsamplingX,
            srcU_ptr + (d->vi.width - border) / d->subsamplingX,
            dstStride, srcStrideY, srcStrideUV, border, d->vi.height, lut_y, lut_u, lut_v);
    }

    if (d->grayscale) {
        const int dstStrideU = vsapi->getStride(dst, 1) / sizeof(pixel_t);
        const int dstStrideV = vsapi->getStride(dst, 2) / sizeof(pixel_t);
        memset(vsapi->getWritePtr(dst, 1), 128, dstStrideU * vsapi->getFrameHeight(dst, 1));
        memset(vsapi->getWritePtr(dst, 2), 128, dstStrideV * vsapi->getFrameHeight(dst, 2));
    }
}

static void VS_CC TCMInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    TCMData * d = static_cast<TCMData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC TCMGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    TCMData * d = static_cast<TCMData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
#ifdef VS_TARGET_CPU_X86
        no_subnormals();
#endif
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef * dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);
        int bits = d->vi.format->bitsPerSample;
        if (d->vi.format->bytesPerSample == 1) {
            process<uint8_t>(src, dst, bits, d->lut_y, d->lut_u, d->lut_v, d, vsapi);
        }
        /*
        no 16 bit implementation for now
        else if (d->vi.format->bytesPerSample == 2) {
            process_c<uint16_t>(src, dst, bits, d->lut_y, d->lut_u, d->lut_v, d, vsapi);
        }*/
        vsapi->freeFrame(src);
        return dst;
    }
    return nullptr;
}

static void VS_CC TCMFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    TCMData * d = static_cast<TCMData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

void VS_CC TCMCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<TCMData> d{ new TCMData{} };
    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);

    try {
        if (d->vi.format->bitsPerSample != 8 || d->vi.format->colorFamily != cmYUV) {
            throw std::string{ "Only 8 bit YUV formats are supported!" };
        }
        std::vector<int> colors;
        int colorsCount = vsapi->propNumElements(in, "colors");
        for (int i = 0; i < colorsCount; i++) {
            colors.push_back(stringToInt(vsapi->propGetData(in, "colors", i, &err)));
        }

        d->tolerance = static_cast<int>(vsapi->propGetInt(in, "tolerance", 0, &err));
        if (err)
            d->tolerance = 10;

        d->bt601 = !!vsapi->propGetInt(in, "bt601", 0, &err);

        d->grayscale = !!vsapi->propGetInt(in, "gray", 0, &err);

        d->prefer_lut_thresh = static_cast<int>(vsapi->propGetInt(in, "lutthr", 0, &err));
        if (err)
            d->prefer_lut_thresh = 9;

        if (d->vi.format->subSamplingW == 1 && d->vi.format->subSamplingH == 1) {
            d->subsamplingX = 2;
            d->subsamplingY = 2;
            d->lutFunction = processLut<2, 2>;
            d->sse2Function = processSse2<2, 2>;
        }
        else if (d->vi.format->subSamplingW == 1 && d->vi.format->subSamplingH == 0) {
            d->subsamplingX = 2;
            d->subsamplingY = 1;
            d->lutFunction = processLut<2, 1>;
            d->sse2Function = processSse2<2, 1>;
        }
        else if (d->vi.format->subSamplingW == 0 && d->vi.format->subSamplingH == 0) {
            d->subsamplingX = 1;
            d->subsamplingY = 1;
            d->lutFunction = processLut<1, 1>;
            d->sse2Function = processSse2<1, 1>;
        }
        
        float kR = d->bt601 ? 0.299f : 0.2126f;
        float kB = d->bt601 ? 0.114f : 0.0722f;

        for (auto color : colors) {
            YUVPixel p;
            memset(&p, 0, sizeof(p));
            float r = static_cast<float>((color & 0xFF0000) >> 16) / 255.0f;
            float g = static_cast<float>((color & 0xFF00) >> 8) / 255.0f;
            float b = static_cast<float>(color & 0xFF) / 255.0f;

            float y = kR*r + (1 - kR - kB)*g + kB*b;
            p.U = 128 + depfree_round(112.0f*(b - y) / (1 - kB));
            p.V = 128 + depfree_round(112.0f*(r - y) / (1 - kR));
            p.Y = 16 + depfree_round(219.0f*y);
            for (int i = 0; i < 4; i++) {
                p.vector_y |= p.Y << (8 * i);
                p.vector_u |= p.U << (8 * i);
                p.vector_v |= p.V << (8 * i);
            }

            d->colors.push_back(p);
        }

        colors.clear();

        for (int i = 0; i < 4; i++) {
            d->vector_tolerance |= d->tolerance << (8 * i);
            d->vector_half_tolerance |= d->vector_half_tolerance | (d->tolerance / 2) << (8 * i);
        }

        if (((d->vi.width % 16) != 0) || (d->colors.size() > d->prefer_lut_thresh)) {
            for (int i = 0; i < 256; ++i) {
                uint8_t val_y = 0;
                uint8_t val_u = 0;
                uint8_t val_v = 0;
                for (auto &color : d->colors) {
                    val_y |= ((abs(i - color.Y) < d->tolerance) ? 255 : 0);
                    val_u |= ((abs(i - color.U) < (d->tolerance / 2)) ? 255 : 0);
                    val_v |= ((abs(i - color.V) < (d->tolerance / 2)) ? 255 : 0);
                }
                d->lut_y[i] = val_y;
                d->lut_u[i] = val_u;
                d->lut_v[i] = val_v;
            }
        }
    }
    catch (const std::string & error) {
        vsapi->setError(out, ("TColorMask: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }
    vsapi->createFilter(in, out, "TColorMask", TCMInit, TCMGetFrame, TCMFree, fmParallel, 0, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.djatom.tcm", "tcm", "TColorMask plugin for VapourSynth.", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("TColorMask",
        "clip:clip;"
        "colors:data[];"
        "tolerance:int:opt;"
        "bt601:int:opt;"
        "gray:int:opt;"
        "lutthr:int:opt;",
        TCMCreate, nullptr, plugin);
}