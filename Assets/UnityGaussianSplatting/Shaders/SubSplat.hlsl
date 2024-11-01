#ifndef SUB_SPLAT_HLSL
#define SUB_SPLAT_HLSL

#include "GaussianSplatting.hlsl"

#pragma Native16Bit

#define SUB_SPLAT_MERGE 0               // Disable explicit merge by default
#define MAX_SUB_SPLAT_LEVEL 4           // at most (1 << MAX_SUB_SPLAT_LEVEL) Sub-Splats can be splitted from a Splat
#define LOG_MAX_SUB_SPLAT_COUNT 18      // kLogMaxSubSplatCount
#define LOG_MAX_SUB_SPLAT_REF_COUNT 18  // kLogMaxSubSplatRefCount
#define LOG_MAX_SUB_SPLAT_INIT_COUNT 17 // kLogMaxSubSplatInitCount
#define MAX_SUB_SPLAT_COUNT (1 << LOG_MAX_SUB_SPLAT_COUNT)
#define MAX_SUB_SPLAT_REF_COUNT (1 << LOG_MAX_SUB_SPLAT_REF_COUNT)
#define MAX_SUB_SPLAT_INIT_COUNT (1 << LOG_MAX_SUB_SPLAT_INIT_COUNT)

#if LOG_MAX_SUB_SPLAT_COUNT > 24
    #error LOG_MAX_SUB_SPLAT_COUNT too large
#endif
#if LOG_MAX_SUB_SPLAT_REF_COUNT < LOG_MAX_SUB_SPLAT_COUNT
    #error LOG_MAX_SUB_SPLAT_REF_COUNT too small
#endif
#if MAX_SUB_SPLAT_LEVEL > 255
    #error MAX_SUB_SPLAT_LEVEL too large
#endif

// Remember to change kGpuSubSplatDataSize in GaussianSplatRenderer.cs if you modify `struct SubSplatData`
struct SubSplatData {
    float3 pos;
    uint rootSplatID;
    uint2 rotFP16x4;
    uint2 scaleOpacityFP16x4;
};

SubSplatData GetSubSplatFromSplat(uint splatID, in const SplatData splat) {
    SubSplatData ret;
    ret.pos = splat.pos;
    ret.rootSplatID = splatID;
    ret.rotFP16x4 = uint2(
        f32tof16(splat.rot.x) | f32tof16(splat.rot.y) << 16, 
        f32tof16(splat.rot.z) | f32tof16(splat.rot.w) << 16
    );
    ret.scaleOpacityFP16x4 = uint2(
        f32tof16(splat.scale.x) | f32tof16(splat.scale.y) << 16, 
        f32tof16(splat.scale.z) | f32tof16(float(splat.opacity)) << 16
    );
    return ret;
}

SplatData GetSplatFromSubSplat(in const SubSplatData subSplat) {
    SplatData ret;
    ret = LoadSplatData(subSplat.rootSplatID);
    ret.pos = subSplat.pos;
    ret.rot = float4(
        f16tof32(subSplat.rotFP16x4.x), f16tof32(subSplat.rotFP16x4.x >> 16),
        f16tof32(subSplat.rotFP16x4.y), f16tof32(subSplat.rotFP16x4.y >> 16)
    );
    ret.scale = float3(
        f16tof32(subSplat.scaleOpacityFP16x4.x), f16tof32(subSplat.scaleOpacityFP16x4.x >> 16),
        f16tof32(subSplat.scaleOpacityFP16x4.y)
    );
    ret.opacity = f16tof32(subSplat.scaleOpacityFP16x4.y >> 16);
    return ret;
}

void SplitSubSplat(in const SubSplatData parentSubSplat, out SubSplatData o_subSplat0, out SubSplatData o_subSplat1) {
    // TODO:
    o_subSplat0 = parentSubSplat;
    o_subSplat1 = parentSubSplat;
}

#if SUB_SPLAT_MERGE == 1
    void MergeSubSplat(in const SubSplatData subSplat0, in const SubSplatData subSplat1, inout SubSplatData io_parentSubSplat) {
        // TODO: Implement this if needed, and remember to set SUB_SPLAT_MERGE to 1
    }
#endif

uint CalcSplatLevel(
    uint splatID, in const SplatData splat, 
    float3 centerWorldPos, float3 centerViewPos, float3 centerClipPos,
    float3x3 rotScaleMat, 
    float3 cov3d0, float3 cov3d1,
    float3 cov2d,
    float2 screenWH
) {
    float d = max(0, -centerViewPos.z);
    return uint(1.0 / d);
}

#endif