// https://github.com/GPUOpen-Effects/FidelityFX-FSR-Unity-URP/blob/master/FSR1.0-For-URP10.6.0-Patch.patch

#pragma kernel KFsrEasuMain
#pragma kernel KFsrEasuInitialize
#pragma kernel KFsrRcasMain
#pragma kernel KFsrRcasInitialize

// #pragma multi_compile _ ENABLE_ALPHA _AMD_FSR_HALF _AMD_FSR_NEEDS_CONVERT_TO_SRGB
// #define ENABLE_ALPHA
// #define _AMD_FSR_HALF

#include "amd_fsr.hlsl"

// 4 elements:
// [0] = const0
// [1] = const1
// [0] = const2
// [1] = const3
// ComputeBuffer is allocated with stride sizeof(int)*4, 4 elements
RWStructuredBuffer<uint4> _EASUParameters;

float4 _EASUViewportSize;
float4 _EASUInputImageSize;
float4 _EASUOutputSize;

SamplerState s_linear_clamp_sampler;
Texture2D<float4> _EASUInputTexture;
RWTexture2D<float4> _EASUOutputTexture;

#ifdef _AMD_FSR_HALF
AH4 FsrEasuRH(AF2 p){ return AH4(AMD_FSR_TO_SRGB(_EASUInputTexture.GatherRed(s_linear_clamp_sampler, p))); }
AH4 FsrEasuGH(AF2 p){ return AH4(AMD_FSR_TO_SRGB(_EASUInputTexture.GatherGreen(s_linear_clamp_sampler, p))); }
AH4 FsrEasuBH(AF2 p){ return AH4(AMD_FSR_TO_SRGB(_EASUInputTexture.GatherBlue(s_linear_clamp_sampler, p))); }
#else
AF4 FsrEasuRF(AF2 p){ return AMD_FSR_TO_SRGB(_EASUInputTexture.GatherRed(s_linear_clamp_sampler, p)); }
AF4 FsrEasuGF(AF2 p){ return AMD_FSR_TO_SRGB(_EASUInputTexture.GatherGreen(s_linear_clamp_sampler, p)); }
AF4 FsrEasuBF(AF2 p){ return AMD_FSR_TO_SRGB(_EASUInputTexture.GatherBlue(s_linear_clamp_sampler, p)); }
#endif

[numthreads(64, 1, 1)]
void KFsrEasuMain(uint3 LocalThreadId : SV_GroupThreadID, uint3 WorkGroupId : SV_GroupID, uint3 dispatchThreadId : SV_DispatchThreadID)
{
    // Do remapping of local xy in workgroup for a more PS-like swizzle pattern.
    AU2 gxy = ARmp8x8(LocalThreadId.x) + AU2(WorkGroupId.x<<3u, WorkGroupId.y<<3u);

#ifdef ENABLE_ALPHA
    AREAL alpha = _EASUInputTexture.Load(int3(gxy.xy, 0)).a;
#else
    AREAL alpha = 1.0;
#endif

    AU4 con0 = _EASUParameters[0];
    AU4 con1 = _EASUParameters[1];
    AU4 con2 = _EASUParameters[2];
    AU4 con3 = _EASUParameters[3];
    AF4 c;

#ifdef _AMD_FSR_HALF
    AH3 out_rgb;
    FsrEasuH(out_rgb, gxy, con0, con1, con2, con3);
#else
    AF3 out_rgb;
    FsrEasuF(out_rgb, gxy, con0, con1, con2, con3);
#endif

    c.rgb = out_rgb;

    c.a = alpha;
    _EASUOutputTexture[gxy] = c;
}


/*
Doing this to avoid having to deal with any CPU side compilation of the headers.
The FsrRcasCon is doing some extra parameter packing (log space / pows etc) so its better
to keep this all in the GPU for simplicity sake, and avoid paying this cost for every wave.
The headers also dont compile for c#, they are meant for c and c++.
*/
[numthreads(1,1,1)]
void KFsrEasuInitialize()
{
    AU4 con0 = (AU4)0;
    AU4 con1 = (AU4)0;
    AU4 con2 = (AU4)0;
    AU4 con3 = (AU4)0;
    FsrEasuCon(con0,con1,con2,con3,
        _EASUViewportSize.x,  _EASUViewportSize.y,
        _EASUInputImageSize.x,_EASUInputImageSize.y,
        _EASUOutputSize.x,    _EASUOutputSize.y);

    _EASUParameters[0] = con0;
    _EASUParameters[1] = con1;
    _EASUParameters[2] = con2;
    _EASUParameters[3] = con3;
}

// one element:
// [0] = const0
// ComputeBuffer is allocated with stride sizeof(int)*4, 1 element
RWStructuredBuffer<uint4> _RCASParameters;

float _RCASScale;

Texture2D<float4> _RCASInputTexture;
RWTexture2D<float4> _RCASOutputTexture;

#ifdef _AMD_FSR_HALF
AH4 FsrRcasLoadH(ASW2 p)  {return AH4(AMD_FSR_TO_SRGB(_RCASInputTexture[p])); }
void FsrRcasInputH(inout AH1 r,inout AH1 g,inout AH1 b) {}
#else
AF4 FsrRcasLoadF(ASU2 p)  {return AMD_FSR_TO_SRGB(_RCASInputTexture[p]); }
void FsrRcasInputF(inout AF1 r,inout AF1 g,inout AF1 b) {}
#endif

void WritePix(AU2 gxy, AF4 casPix)
{
    _RCASOutputTexture[gxy] = casPix;
}


[numthreads(64, 1, 1)]
void KFsrRcasMain(uint3 LocalThreadId : SV_GroupThreadID, uint3 WorkGroupId : SV_GroupID, uint3 dispatchThreadId : SV_DispatchThreadID)
{
    // Do remapping of local xy in workgroup for a more PS-like swizzle pattern.
    AU2 gxy = ARmp8x8(LocalThreadId.x) + AU2(WorkGroupId.x << 3u, WorkGroupId.y << 3u);
#ifdef ENABLE_ALPHA
    AREAL alpha = _RCASInputTexture.Load(int3(gxy.xy, 0)).a;
#else
    AREAL alpha = 1.0;
#endif


    AU4 con = _RCASParameters[0];
#ifdef _AMD_FSR_HALF
    AH4 c;
    FsrRcasH(c.r, c.g, c.b, gxy, con);
#else
    AF4 c;
    FsrRcasF(c.r, c.g, c.b, gxy, con);
#endif

    c.a = alpha;
    WritePix(gxy, c);
}


/*
Doing this to avoid having to deal with any CPU side compilation of the headers.
The FsrRcasCon is doing some extra parameter packing (log space / pows etc) so its better
to keep this all in the GPU for simplicity sake, and avoid paying this cost for every wave.
The headers also dont compile for c#, they are meant for c and c++.
*/
[numthreads(1,1,1)]
void KFsrRcasInitialize()
{
    AU4 con;
    FsrRcasCon(con, _RCASScale);
    _RCASParameters[0] = con;
}