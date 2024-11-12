#ifndef SUB_SPLAT_HLSL
#define SUB_SPLAT_HLSL

#include "GaussianSplatting.hlsl"

#define SUB_SPLAT_MERGE 0               // Disable explicit merge by default
#define MAX_SUB_SPLAT_LEVEL 6           // at most (1 << MAX_SUB_SPLAT_LEVEL) Sub-Splats can be splitted from a Splat
#define LOG_MAX_SUB_SPLAT_COUNT 18      // kLogMaxSubSplatCount
#define MAX_SUB_SPLAT_COUNT (1 << LOG_MAX_SUB_SPLAT_COUNT)

#if LOG_MAX_SUB_SPLAT_COUNT > 24
    #error LOG_MAX_SUB_SPLAT_COUNT too large
#endif
#if MAX_SUB_SPLAT_LEVEL > 15
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
    UnpackedSubSplatData subSplat0 = parentSubSplat, subSplat1 = parentSubSplat;
    float3 pos = parentSubSplat.pos;
    float3 scale = parentSubSplat.scale;
    float opacity = parentSubSplat.opacity;
    float4 rot = parentSubSplat.rot;
    float3 cov3d0, cov3d1;
    CalcCovariance3D(CalcMatrixFromRotationScale(rot, scale), cov3d0, cov3d1);
    float3x3 covMatrix = float3x3(
        cov3d0.x, cov3d0.y, cov3d0.z,
        cov3d0.y, cov3d1.x, cov3d1.y,
        cov3d0.z, cov3d1.y, cov3d1.z
    );
    float PI = 3.14159265359f;
    float epsilon = 1e-20f;
    
    float C = 0.5f + epsilon;
    float C2 = C * C;
    float sqrt2PI = sqrt(2.0f * PI);
    float D = 1.0f / sqrt2PI;

    float3x3 mr = CalcMatrixFromRotationScale(rot, float3(1, 1, 1));
    /* float3 mainAxisLocal = float3(step(scale.y, scale.x) * step(scale.z, scale.x), step(scale.x, scale.y) * step(scale.z, scale.y), step(scale.x, scale.z) * step(scale.y, scale.z));
    float3 mainAxisWorld = normalize(mul(mr, mainAxisLocal)); */
    float3 mainAxisWorld;
    if (scale.x > scale.y && scale.x > scale.z)
        mainAxisWorld = float3(mr[0][0], mr[1][0], mr[2][0]);
    else if (scale.y > scale.z)
        mainAxisWorld = float3(mr[0][1], mr[1][1], mr[2][1]);
    else
        mainAxisWorld = float3(mr[0][2], mr[1][2], mr[2][2]);
    
    float3 covMulAxisWorld = mul(covMatrix, mainAxisWorld);
    float tau = sqrt(dot(mainAxisWorld, covMulAxisWorld));
    float tau2 = tau * tau;
    float3 L0 = mul(covMatrix, mainAxisWorld);
    
    float D2_C2 = D * D / C2;
    float3x3 L02 = float3x3(
        L0.x * L0.x, L0.x * L0.y, L0.x * L0.z,
        L0.y * L0.x, L0.y * L0.y, L0.y * L0.z,
        L0.z * L0.x, L0.z * L0.y, L0.z * L0.z
    );

    float3x3 L02D2_tau2C2 = L02 * D2_C2 / tau2;
    float3 L0D_tauC = L0 * D / tau / C;

    // float subSplatOpacity = opacity * 0.5f;
    float subSplatOpacity = opacity * 0.9; // Incorrect in theory, but has better effect
    float3 subSplatPos0 = pos - L0D_tauC;
    float3 subSplatPos1 = pos + L0D_tauC;

    float3x3 subsplatCovMat = covMatrix - L02D2_tau2C2;
    float3 subSplatCov3d0 = float3(subsplatCovMat[0][0], subsplatCovMat[0][1], subsplatCovMat[0][2]);
    float3 subSplatCov3d1 = float3(subsplatCovMat[1][1], subsplatCovMat[1][2], subsplatCovMat[2][2]);

    subSplat0.pos = subSplatPos0;
    subSplat1.pos = subSplatPos1;
    subSplat0.opacity = subSplatOpacity;
    subSplat1.opacity = subSplatOpacity;
    
    float3 subSplatScale;
    float4 subSplatRot;
    DecomposeCovariance3D(subSplatCov3d0, subSplatCov3d1, subSplatScale, subSplatRot, 8, false);

    subSplat0.scale = subSplatScale;
    subSplat1.scale = subSplatScale;
    subSplat0.rot = subSplatRot;
    subSplat1.rot = subSplatRot;
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
    float lambda1, _;
    DecomposeCovarianceLambda(cov2d, lambda1, _);
    float scale = sqrt(lambda1);
    float k = 4.0 * scale / max(screenWH.x, screenWH.y);
    const float k0 = 0.05;
    float r = max(k / k0, 1.0);
    uint level = (asuint(r) >> 23) - 127; // log_2
    return level;
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
