// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            Blend OneMinusDstAlpha One
            Cull Off
            
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc

#include "GaussianSplatting.hlsl"

StructuredBuffer<uint> _OrderBuffer;

struct v2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float4 vertex : SV_POSITION;
};

StructuredBuffer<SplatViewData> _SplatViewData;
ByteAddressBuffer _SplatSelectedBits;
uint _SplatBitsValid;

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o = (v2f)0;
    instID = _OrderBuffer[instID];
	SplatViewData view = _SplatViewData[instID];
    float2 centerClipXY = float2(f16tof32(view.clipXY), f16tof32(view.clipXY >> 16u));
	float2 axis1 = float2(f16tof32(view.axis.x), f16tof32(view.axis.x >> 16u));
	float2 axis2 = float2(f16tof32(view.axis.y), f16tof32(view.axis.y >> 16u));

	o.col.r = f16tof32(view.color.x >> 16);
	o.col.g = f16tof32(view.color.x);
	o.col.b = f16tof32(view.color.y >> 16);
	o.col.a = f16tof32(view.color.y);

	uint idx = vtxID;
	float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
	quadPos *= 2;

	o.pos = quadPos;

	float2 deltaScreenPos = (quadPos.x * axis1 + quadPos.y * axis2) * 2 / _ScreenParams.xy;
	o.vertex = float4(centerClipXY, 0, 1);
	o.vertex.xy += deltaScreenPos;
	
	// is this splat selected?
	if (_SplatBitsValid)
	{
		uint splatID = view.subSplatFlagSplatID & 0x7FFFFFFF;
		bool isSubSplat = bool(view.subSplatFlagSplatID >> 31);

		uint wordIdx = splatID / 32;
		uint bitIdx = splatID & 31;
		uint selVal = _SplatSelectedBits.Load(wordIdx * 4);
		if (selVal & (1 << bitIdx))
			o.col.a = isSubSplat ? -2 : -1;
	}

    return o;
}

half4 frag (v2f i) : SV_Target
{
	float power = -dot(i.pos, i.pos);
	half alpha = exp(power);
	if (i.col.a >= 0)
	{
		alpha = saturate(alpha * i.col.a);
	}
	else
	{
		// "selected" splat: magenta outline, increase opacity, magenta tint
		half3 selectedColor = i.col.a == -1 ? half3(1,0,1) : half3(0,0,1);
		if (alpha > 7.0/255.0)
		{
			if (alpha < 10.0/255.0)
			{
				alpha = 1;
				i.col.rgb = selectedColor;
			}
			alpha = saturate(alpha + 0.3);
		}
		i.col.rgb = lerp(i.col.rgb, selectedColor, 0.5);
	}
	
    if (alpha < 1.0/255.0)
        discard;

    half4 res = half4(i.col.rgb * alpha, alpha);
    return res;
}
ENDCG
        }
    }
}
