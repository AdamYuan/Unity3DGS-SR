using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

namespace GaussianSplatting.Runtime
{
    // GPU (uint key, uint payload) 8 bit-LSD radix sort, using reduce-then-scan
    // Copyright Thomas Smith 2023, MIT license
    // https://github.com/b0nes164/GPUSorting

    public class GpuSorting
    {
        protected const int k_radix = 256;
        protected const int k_radixPasses = 4;
        protected const int k_partitionSize = 3840;

        protected const int k_minSize = 1;
        protected const int k_maxSize = 65535 * k_partitionSize;

        protected const int k_globalHistPartSize = 32768;

        public struct Args
        {
            public uint count;
            public GraphicsBuffer inputKeys;
            public GraphicsBuffer inputValues;
            public SupportResources resources;
            internal int workGroupCount;
        }

        public struct SupportResources
        {
            public GraphicsBuffer tempKeyBuffer;
            public GraphicsBuffer tempPayloadBuffer;
            public GraphicsBuffer tempGlobalHistBuffer;
            public GraphicsBuffer tempPassHistBuffer;
            public GraphicsBuffer tempIndexBuffer;

            public static SupportResources Load(uint count)
            {
                var target = GraphicsBuffer.Target.Structured;
                var resources = new SupportResources
                {
                    tempKeyBuffer = new GraphicsBuffer(target, (int)count, 4) { name = "OneSweepAltKey" },
                    tempPayloadBuffer = new GraphicsBuffer(target, (int)count, 4) { name = "OneSweepAltPayload" },
                    tempGlobalHistBuffer = new GraphicsBuffer(target, k_radix * k_radixPasses, 4) { name = "OneSweepGlobalHistogram" },
                    tempPassHistBuffer = new GraphicsBuffer(target, (int)(k_radix * DivRoundUp(count, k_partitionSize) * k_radixPasses), 4) { name = "OneSweepPassHistogram" },
                    tempIndexBuffer = new GraphicsBuffer(target, k_radixPasses, sizeof(uint)) { name = "OneSweepIndex" },
                };
                return resources;
            }

            public void Dispose()
            {
                tempKeyBuffer?.Dispose();
                tempPayloadBuffer?.Dispose();
                tempGlobalHistBuffer?.Dispose();
                tempPassHistBuffer?.Dispose();
                tempIndexBuffer?.Dispose();

                tempKeyBuffer = null;
                tempPayloadBuffer = null;
                tempGlobalHistBuffer = null;
                tempPassHistBuffer = null;
                tempIndexBuffer = null;
            }
        }

        readonly ComputeShader m_CS;
        protected int m_kernelInit = -1;
        protected int m_kernelGlobalHist = -1;
        protected int m_kernelScan = -1;
        protected int m_digitBinningPass = -1;

        readonly bool m_Valid;

        public bool Valid => m_Valid;

        public GpuSorting(ComputeShader cs)
        {
            m_CS = cs;
            if (cs)
            {
                m_kernelInit = m_CS.FindKernel("InitSweep");
                m_kernelGlobalHist = m_CS.FindKernel("GlobalHistogram");
                m_kernelScan = m_CS.FindKernel("Scan");
                m_digitBinningPass = m_CS.FindKernel("DigitBinningPass");
            }

            m_Valid = m_kernelInit >= 0 &&
                        m_kernelGlobalHist >= 0 &&
                        m_kernelScan >= 0 &&
                        m_digitBinningPass >= 0;

            if (m_Valid)
            {
                if (!m_CS.IsSupported(m_kernelInit) ||
                    !m_CS.IsSupported(m_kernelGlobalHist) ||
                    !m_CS.IsSupported(m_kernelScan) ||
                    !m_CS.IsSupported(m_digitBinningPass))
                {
                    m_Valid = false;
                }
            }
        }

        static uint DivRoundUp(uint x, uint y) => (x + y - 1) / y;

        //Can we remove the last 4 padding without breaking?
        struct SortConstants
        {
            public uint numKeys;                        // The number of keys to sort
            public uint radixShift;                     // The radix shift value for the current pass
            public uint threadBlocks;                   // threadBlocks
            public uint padding0;                       // Padding - unused
        }

        private void SetStaticRootParameters(
            int numKeys,
            CommandBuffer cmd,
            GraphicsBuffer sortBuffer,
            GraphicsBuffer passHistBuffer,
            GraphicsBuffer globalHistBuffer,
            GraphicsBuffer indexBuffer)
        {
            cmd.SetComputeIntParam(m_CS, "e_numKeys", numKeys);

            cmd.SetComputeBufferParam(m_CS, m_kernelInit, "b_passHist", passHistBuffer);
            cmd.SetComputeBufferParam(m_CS, m_kernelInit, "b_globalHist", globalHistBuffer);
            cmd.SetComputeBufferParam(m_CS, m_kernelInit, "b_index", indexBuffer);

            cmd.SetComputeBufferParam(m_CS, m_kernelGlobalHist, "b_sort", sortBuffer);
            cmd.SetComputeBufferParam(m_CS, m_kernelGlobalHist, "b_globalHist", globalHistBuffer);

            cmd.SetComputeBufferParam(m_CS, m_kernelScan, "b_passHist", passHistBuffer);
            cmd.SetComputeBufferParam(m_CS, m_kernelScan, "b_globalHist", globalHistBuffer);

            cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_passHist", passHistBuffer);
            cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_index", indexBuffer);
        }


        private void Dispatch(
            int numThreadBlocks,
            int globalHistThreadBlocks,
            CommandBuffer cmd,
            GraphicsBuffer toSort,
            GraphicsBuffer toSortPayload,
            GraphicsBuffer alt,
            GraphicsBuffer altPayload)
        {
            cmd.SetComputeIntParam(m_CS, "e_threadBlocks", numThreadBlocks);
            cmd.DispatchCompute(m_CS, m_kernelInit, 256, 1, 1);

            cmd.SetComputeIntParam(m_CS, "e_threadBlocks", globalHistThreadBlocks);
            cmd.DispatchCompute(m_CS, m_kernelGlobalHist, globalHistThreadBlocks, 1, 1);

            cmd.SetComputeIntParam(m_CS, "e_threadBlocks", numThreadBlocks);
            cmd.DispatchCompute(m_CS, m_kernelScan, k_radixPasses, 1, 1);
            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                cmd.SetComputeIntParam(m_CS, "e_radixShift", radixShift);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_sort", toSort);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_sortPayload", toSortPayload);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_alt", alt);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_altPayload", altPayload);
                cmd.DispatchCompute(m_CS, m_digitBinningPass, numThreadBlocks, 1, 1);

                (toSort, alt) = (alt, toSort);
                (toSortPayload, altPayload) = (altPayload, toSortPayload);
            }
        }

        public void Dispatch(CommandBuffer cmd, Args args)
        {
            Assert.IsTrue(Valid);

            int threadBlocks = (int)DivRoundUp(args.count, k_partitionSize);
            int globalHistThreadBlocks = (int)DivRoundUp(args.count, k_globalHistPartSize);

            SetStaticRootParameters(
                (int)args.count,
                cmd,
                args.inputKeys,
                args.resources.tempPassHistBuffer,
                args.resources.tempGlobalHistBuffer,
                args.resources.tempIndexBuffer);

            Dispatch(
                threadBlocks,
                globalHistThreadBlocks,
                cmd,
                args.inputKeys,
                args.inputValues,
                args.resources.tempKeyBuffer,
                args.resources.tempPayloadBuffer);
        }
    }
}
