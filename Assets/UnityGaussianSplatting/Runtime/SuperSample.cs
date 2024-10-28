using UnityEngine;
using UnityEngine.Rendering;

namespace GaussianSplatting.Runtime
{
    public class SuperSample
    {
        readonly ComputeShader m_CS;
        protected int m_kernelFsrEasuInit = -1;
        protected int m_kernelFsrEasuMain = -1;
        protected int m_kernelFsrRcasInit = -1;
        protected int m_kernelFsrRcasMain = -1;
        protected int m_fsrTmpID = Shader.PropertyToID("_FsrTmpTexture");

        public struct SupportResources
        {
            public GraphicsBuffer easuParamBuffer;
            public GraphicsBuffer rcasParamBuffer;

            public static SupportResources Load()
            {
                var target = GraphicsBuffer.Target.Structured;
                var resources = new SupportResources
                {
                    easuParamBuffer = new GraphicsBuffer(target, 4, 4 * sizeof(uint)) { name = "FsrEasuParam" },
                    rcasParamBuffer = new GraphicsBuffer(target, 1, 4 * sizeof(uint)) { name = "FsrRcasParam" },
                };
                return resources;
            }

            public void Dispose()
            {
                easuParamBuffer?.Dispose();
                rcasParamBuffer?.Dispose();
                
                easuParamBuffer = null;
                rcasParamBuffer = null;
            }
        }

        public struct Args
        {
            public Vector2Int srcSize, dstSize;
            public RTHandle target, tmp;
            // target and tmp's sizes should equals to dstSize, tmp can be null
            public SupportResources resources;
        }

        public SuperSample(ComputeShader cs)
        {
            m_CS = cs;
            m_kernelFsrEasuInit = m_CS.FindKernel("KFsrEasuInitialize");
            m_kernelFsrEasuMain = m_CS.FindKernel("KFsrEasuMain");
            m_kernelFsrRcasInit = m_CS.FindKernel("KFsrRcasInitialize");
            m_kernelFsrRcasMain = m_CS.FindKernel("KFsrRcasMain");
        }

        private RenderTextureDescriptor GetUAVCompatibleDescriptor(RTHandle target, int width, int height)
        {
            RenderTextureDescriptor desc = target.rt.descriptor;
            desc.depthBufferBits = 0;
            desc.msaaSamples = 1;
            desc.width = width;
            desc.height = height;
            desc.enableRandomWrite = true;
            desc.mipCount = 1;
            return desc;
        }

        public void DispatchFSR(CommandBuffer cmd, Args args)
        {
            if (args.resources.easuParamBuffer == null || args.resources.rcasParamBuffer == null || args.target == null) {
                Debug.Log("Why?");
                return;
            }

            int groupX = (args.dstSize.x + 7) / 8;
            int groupY = (args.dstSize.y + 7) / 8;

            // Global Params
            cmd.SetComputeVectorParam(m_CS, "_EASUViewportSize", new Vector4(args.srcSize.x, args.srcSize.y));
            cmd.SetComputeVectorParam(m_CS, "_EASUInputImageSize", new Vector4(args.dstSize.x, args.dstSize.y));
            cmd.SetComputeVectorParam(m_CS, "_EASUOutputSize", new Vector4(args.dstSize.x, args.dstSize.y, 1.0f / args.dstSize.x, 1.0f / args.dstSize.y));
            cmd.SetComputeFloatParam(m_CS, "_RCASScale", 1.0f);

            // Alloc TemporaryRT
            if (args.tmp == null)
                cmd.GetTemporaryRT(m_fsrTmpID, GetUAVCompatibleDescriptor(args.target, args.dstSize.x, args.dstSize.y));

            // EASU Init
            cmd.SetComputeBufferParam(m_CS, m_kernelFsrEasuInit, "_EASUParameters", args.resources.easuParamBuffer);
            cmd.DispatchCompute(m_CS, m_kernelFsrEasuInit, 1, 1, 1);
            
            // EASU Main
            cmd.SetComputeBufferParam(m_CS, m_kernelFsrEasuMain, "_EASUParameters", args.resources.easuParamBuffer);
            cmd.SetComputeTextureParam(m_CS, m_kernelFsrEasuMain, "_EASUInputTexture", args.target);
            if (args.tmp == null)
                cmd.SetComputeTextureParam(m_CS, m_kernelFsrEasuMain, "_EASUOutputTexture", m_fsrTmpID);
            else
                cmd.SetComputeTextureParam(m_CS, m_kernelFsrEasuMain, "_EASUOutputTexture", args.tmp);
            cmd.DispatchCompute(m_CS, m_kernelFsrEasuMain, groupX, groupY, 1);

            // RCAS Init
            cmd.SetComputeBufferParam(m_CS, m_kernelFsrRcasInit, "_RCASParameters", args.resources.rcasParamBuffer);
            cmd.DispatchCompute(m_CS, m_kernelFsrRcasInit, 1, 1, 1);

            // RCAS Main
            cmd.SetComputeBufferParam(m_CS, m_kernelFsrRcasMain, "_RCASParameters", args.resources.rcasParamBuffer);
            if (args.tmp == null)
                cmd.SetComputeTextureParam(m_CS, m_kernelFsrRcasMain, "_RCASInputTexture", m_fsrTmpID);
            else
                cmd.SetComputeTextureParam(m_CS, m_kernelFsrRcasMain, "_RCASInputTexture", args.tmp);
            cmd.SetComputeTextureParam(m_CS, m_kernelFsrRcasMain, "_RCASOutputTexture", args.target);
            cmd.DispatchCompute(m_CS, m_kernelFsrRcasMain, groupX, groupY, 1);
            
            // Free TemporaryRT
            if (args.tmp == null)
                cmd.ReleaseTemporaryRT(m_fsrTmpID);
        }
    }
}
