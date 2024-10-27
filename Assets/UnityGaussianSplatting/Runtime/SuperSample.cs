using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

namespace GaussianSplatting.Runtime
{
    // GPU (uint key, uint payload) 8 bit-LSD radix sort, using reduce-then-scan
    // Copyright Thomas Smith 2023, MIT license
    // https://github.com/b0nes164/GPUSorting

    public class SuperSample
    {
        readonly ComputeShader m_CS;
        protected int m_kernelFSR_EASU = -1;
        protected int m_kernelFSR_RCAS = -1;

        public SuperSample(ComputeShader cs)
        {
            m_CS = cs;
            m_kernelFSR_EASU = m_CS.FindKernel("FSR_EASU");
            m_kernelFSR_RCAS = m_CS.FindKernel("FSR_RCAS");
        }

        static uint DivRoundUp(uint x, uint y) => (x + y - 1) / y;

        public void DispatchFSR(CommandBuffer cmd, RTHandle target, RTHandle tmp, Vector2Int srcSize, Vector2Int dstSize)
        {
        }
    }
}
