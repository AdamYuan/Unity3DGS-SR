#ifndef SUB_SPLAT_HLSL
#define SUB_SPLAT_HLSL

#include "GaussianSplatting.hlsl"

#define SUB_SPLAT_MERGE 0               // Disable explicit merge by default
#define MAX_SUB_SPLAT_LEVEL 6           // at most (1 << MAX_SUB_SPLAT_LEVEL) Sub-Splats can be splitted from a Splat
#define LOG_MAX_SUB_SPLAT_COUNT 19      // kLogMaxSubSplatCount
#define MAX_SUB_SPLAT_COUNT (1 << LOG_MAX_SUB_SPLAT_COUNT)

#if LOG_MAX_SUB_SPLAT_COUNT > 24
    #error LOG_MAX_SUB_SPLAT_COUNT too large
#endif
#if MAX_SUB_SPLAT_LEVEL > 255
    #error MAX_SUB_SPLAT_LEVEL too large
#endif

#define __PUBLIC__
// Interface are marked with __PUBLIC__

// Remember to change kGpuSubSplatDataSize in GaussianSplatRenderer.cs if you modify `struct SubSplatData`
__PUBLIC__ struct SubSplatData {
    float3 pos;
    uint2 rotFP16x4;
    uint2 scaleOpacityFP16x4;
};

// A unpacked version of struct SubSplatData, not necessary
struct UnpackedSubSplatData {
    float3 pos;
    float4 rot;
    float3 scale;
    float opacity;
};

UnpackedSubSplatData GetUnpackedSubSplatFromSplat(uint splatID, in const SplatData splat) {
    UnpackedSubSplatData ret;
    ret.pos = splat.pos;
    ret.rot = splat.rot;
    ret.scale = splat.scale;
    ret.opacity = splat.opacity;
    return ret;
}

UnpackedSubSplatData UnpackSubSplatData(in const SubSplatData subSplat) {
    UnpackedSubSplatData ret;
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

SubSplatData PackSubSplatData(in const UnpackedSubSplatData subSplat) {
    SubSplatData ret;
    ret.pos = subSplat.pos;
    ret.rotFP16x4 = uint2(
        f32tof16(subSplat.rot.x) | f32tof16(subSplat.rot.y) << 16, 
        f32tof16(subSplat.rot.z) | f32tof16(subSplat.rot.w) << 16
    );
    ret.scaleOpacityFP16x4 = uint2(
        f32tof16(subSplat.scale.x) | f32tof16(subSplat.scale.y) << 16, 
        f32tof16(subSplat.scale.z) | f32tof16(subSplat.opacity) << 16
    );
    return ret;
}

void SplitUnpackedSubSplat(
    uint rootSplatID,
    uint parentLevel,
    in const UnpackedSubSplatData parentSubSplat, 
    out UnpackedSubSplatData o_subSplat0, 
    out UnpackedSubSplatData o_subSplat1
) {
    // TODO: Implement your splitting behavior
    UnpackedSubSplatData subSplat0 = parentSubSplat, subSplat1 = parentSubSplat;
    float3 scale = parentSubSplat.scale * 0.5;
    subSplat0.scale = scale;
    subSplat0.pos += scale;
    subSplat1.scale = scale;
    subSplat1.pos -= scale;

    o_subSplat0 = subSplat0;
    o_subSplat1 = subSplat1;
}

void MergeUnpackedSubSplat(
    uint rootSplatID,
    in const UnpackedSubSplatData subSplat0, 
    in const UnpackedSubSplatData subSplat1, 
    uint parentLevel,
    inout UnpackedSubSplatData io_parentSubSplat
) {
    // TODO: Implement this if needed, and remember to set SUB_SPLAT_MERGE to 1
}

__PUBLIC__ uint CalcSplatLevel(
    uint splatID, in const SplatData splat, 
    float3 centerWorldPos, float3 centerViewPos, float3 centerClipPos,
    float3x3 rotScaleMat, 
    float3 cov3d0, float3 cov3d1,
    float3 cov2d,
    float2 screenWH
) {
    // TODO: Implement your splitting level behavior
    float d = max(0, -centerViewPos.z);
    return uint(1.0 / d);
}

__PUBLIC__ SplatData GetSplatFromSubSplat(uint rootSplatID, in const SubSplatData subSplat) {
    UnpackedSubSplatData unpackedSubSplat = UnpackSubSplatData(subSplat);
    SplatData ret;
    ret = LoadSplatData(rootSplatID); // Load SH
    ret.pos = unpackedSubSplat.pos;
    ret.rot = unpackedSubSplat.rot;
    ret.scale = unpackedSubSplat.scale;
    ret.opacity = unpackedSubSplat.opacity;
    return ret;
}

__PUBLIC__ void InitSubSplat(uint rootSplatID, in const SplatData rootSplat, out SubSplatData o_subSplat0, out SubSplatData o_subSplat1) {
    UnpackedSubSplatData unpackedRootSubSplat = GetUnpackedSubSplatFromSplat(rootSplatID, rootSplat);
    UnpackedSubSplatData unpackedSubSplat0, unpackedSubSplat1;
    SplitUnpackedSubSplat(rootSplatID, 0, unpackedRootSubSplat, unpackedSubSplat0, unpackedSubSplat1); // root is level 0
    o_subSplat0 = PackSubSplatData(unpackedSubSplat0);
    o_subSplat1 = PackSubSplatData(unpackedSubSplat1);
}

__PUBLIC__ void SplitSubSplat(uint rootSplatID, uint parentLevel, in const SubSplatData parentSubSplat, out SubSplatData o_subSplat0, out SubSplatData o_subSplat1) {
    UnpackedSubSplatData unpackedParentSubSplat = UnpackSubSplatData(parentSubSplat);
    UnpackedSubSplatData unpackedSubSplat0, unpackedSubSplat1;
    SplitUnpackedSubSplat(rootSplatID, parentLevel, unpackedParentSubSplat, unpackedSubSplat0, unpackedSubSplat1);
    o_subSplat0 = PackSubSplatData(unpackedSubSplat0);
    o_subSplat1 = PackSubSplatData(unpackedSubSplat1);
}

#if SUB_SPLAT_MERGE == 1
    __PUBLIC__ void MergeSubSplat(uint rootSplatID, in const SubSplatData subSplat0, in const SubSplatData subSplat1, uint parentLevel, inout SubSplatData io_parentSubSplat) {
        UnpackedSubSplatData unpackedSubSplat0 = UnpackSubSplatData(subSplat0);
        UnpackedSubSplatData unpackedSubSplat1 = UnpackSubSplatData(subSplat1);
        UnpackedSubSplatData unpackedParentSubSplat = UnpackSubSplatData(io_parentSubSplat);
        MergeUnpackedSubSplat(rootSplatID, unpackedSubSplat0, unpackedSubSplat1, parentLevel, unpackedParentSubSplat);
        io_parentSubSplat = PackSubSplatData(unpackedParentSubSplat);
    }
#endif

#undef __PUBLIC__

#endif