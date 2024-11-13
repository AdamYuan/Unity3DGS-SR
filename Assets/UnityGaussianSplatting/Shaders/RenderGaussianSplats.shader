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
#pragma geometry geom
#pragma fragment frag
#pragma require compute
#pragma use_dxc

#include "GaussianSplatting.hlsl"

StructuredBuffer<uint> _OrderBuffer;

struct v2g {
    half4 col : COLOR0;
	half4 axis : TEXCOORD0;
	float2 center : TEXCOORD1;
};

struct g2f {
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float4 vertex : SV_POSITION;
};

StructuredBuffer<SplatViewData> _SplatViewData;
ByteAddressBuffer _SplatSelectedBits;
uint _SplatBitsValid;

v2g vert(uint instID : SV_InstanceID)
{
	v2g o = (v2g)0;
	SplatViewData view = _SplatViewData[_OrderBuffer[instID]];
    float2 centerClipXY = float2(f16tof32(view.clipXY), f16tof32(view.clipXY >> 16u));
	float2 axis1 = float2(f16tof32(view.axis.x), f16tof32(view.axis.x >> 16u));
	float2 axis2 = float2(f16tof32(view.axis.y), f16tof32(view.axis.y >> 16u));

	o.col.r = f16tof32(view.color.x >> 16);
	o.col.g = f16tof32(view.color.x);
	o.col.b = f16tof32(view.color.y >> 16);
	o.col.a = f16tof32(view.color.y);
	
	o.axis = half4(axis1, axis2);
	o.center = centerClipXY;

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

[maxvertexcount(4)]
void geom(point v2g points[1], inout TriangleStream<g2f> tri) {
	v2g p = points[0];
	
	g2f o;
	o.col = p.col;

	o.pos = float2(-2, -2);
	o.vertex = float4(p.center + (o.pos.x * p.axis.xy + o.pos.y * p.axis.zw) * 2.0 / _ScreenParams.xy, 0, 1);
	tri.Append(o);

	o.pos = float2(2, -2);
	o.vertex = float4(p.center + (o.pos.x * p.axis.xy + o.pos.y * p.axis.zw) * 2.0 / _ScreenParams.xy, 0, 1);
	tri.Append(o);

	o.pos = float2(-2, 2);
	o.vertex = float4(p.center + (o.pos.x * p.axis.xy + o.pos.y * p.axis.zw) * 2.0 / _ScreenParams.xy, 0, 1);
	tri.Append(o);

	o.pos = float2(2, 2);
	o.vertex = float4(p.center + (o.pos.x * p.axis.xy + o.pos.y * p.axis.zw) * 2.0 / _ScreenParams.xy, 0, 1);
	tri.Append(o);
}

half4 frag (g2f i) : SV_Target
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
