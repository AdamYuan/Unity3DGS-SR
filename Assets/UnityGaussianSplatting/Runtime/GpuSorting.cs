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
        }

        public struct IndirectArgs
        {
            public GraphicsBuffer count;
            public GraphicsBuffer inputKeys;
            public GraphicsBuffer inputValues;
            public SupportResources resources;
        }

        public struct SupportResources
        {
            public GraphicsBuffer tempKeyBuffer;
            public GraphicsBuffer tempPayloadBuffer;
            public GraphicsBuffer tempGlobalHistBuffer;
            public GraphicsBuffer tempPassHistBuffer;
            public GraphicsBuffer tempIndexBuffer;
            public GraphicsBuffer histIndirectBuffer, sortIndirectBuffer, copyIndirectBuffer;

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

            public static SupportResources LoadIndirect(uint max_count)
            {
                var target = GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments | GraphicsBuffer.Target.Constant;
                var resources = Load(max_count);
                resources.histIndirectBuffer = new GraphicsBuffer(target, 3, sizeof(uint));
                resources.sortIndirectBuffer = new GraphicsBuffer(target, 3, sizeof(uint));
                resources.copyIndirectBuffer = new GraphicsBuffer(target, 3, sizeof(uint));
                return resources;
            }

            public void Dispose()
            {
                tempKeyBuffer?.Dispose();
                tempPayloadBuffer?.Dispose();
                tempGlobalHistBuffer?.Dispose();
                tempPassHistBuffer?.Dispose();
                tempIndexBuffer?.Dispose();
                histIndirectBuffer?.Dispose();
                sortIndirectBuffer?.Dispose();
                copyIndirectBuffer?.Dispose();

                tempKeyBuffer = null;
                tempPayloadBuffer = null;
                tempGlobalHistBuffer = null;
                tempPassHistBuffer = null;
                tempIndexBuffer = null;
                histIndirectBuffer = null;
                sortIndirectBuffer = null;
                copyIndirectBuffer = null;
            }
        }

        readonly ComputeShader m_CS;
        protected int m_kernelInit = -1;
        protected int m_kernelGlobalHist = -1;
        protected int m_kernelScan = -1;
        protected int m_digitBinningPass = -1;
        protected int m_kernelInitIndirect = -1;
        protected int m_kernelCopy = -1;
        protected LocalKeyword m_indirectKeyword;

        readonly bool m_Valid;

        public bool Valid => m_Valid;

        public GpuSorting(ComputeShader cs)
        {
            m_CS = cs;
            if (cs)
            {
                m_kernelInitIndirect = m_CS.FindKernel("InitIndirect");
                m_kernelInit = m_CS.FindKernel("InitSweep");
                m_kernelGlobalHist = m_CS.FindKernel("GlobalHistogram");
                m_kernelScan = m_CS.FindKernel("Scan");
                m_digitBinningPass = m_CS.FindKernel("DigitBinningPass");
                m_kernelCopy = m_CS.FindKernel("Copy");
                
                m_indirectKeyword = new LocalKeyword(cs, "SORT_INDIRECT");
            }

            m_Valid = m_kernelInitIndirect >= 0 && 
                        m_kernelInit >= 0 &&
                        m_kernelGlobalHist >= 0 &&
                        m_kernelScan >= 0 &&
                        m_digitBinningPass >= 0 &&
                        m_kernelCopy >= 0;

            if (m_Valid)
            {
                if (!m_CS.IsSupported(m_kernelInitIndirect) ||
                    !m_CS.IsSupported(m_kernelInit) ||
                    !m_CS.IsSupported(m_kernelGlobalHist) ||
                    !m_CS.IsSupported(m_kernelScan) ||
                    !m_CS.IsSupported(m_digitBinningPass) ||
                    !m_CS.IsSupported(m_kernelCopy))
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
            if (numKeys >= 0)
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

        private void DispatchIndirect(
            GraphicsBuffer histIndirect,
            GraphicsBuffer sortIndirect,
            GraphicsBuffer copyIndirect,
            GraphicsBuffer numHistThreadBlocks,
            GraphicsBuffer numSortThreadBlocks,
            CommandBuffer cmd,
            GraphicsBuffer toSort,
            GraphicsBuffer toSortPayload,
            GraphicsBuffer alt,
            GraphicsBuffer altPayload,
            int bits,
            bool copyIfFlip)
        {
            cmd.SetComputeConstantBufferParam(m_CS, "cbGpuSortingNumThreadBlocks", numSortThreadBlocks, 0, sizeof(uint));
            cmd.DispatchCompute(m_CS, m_kernelInit, 256, 1, 1);

            cmd.SetComputeConstantBufferParam(m_CS, "cbGpuSortingNumThreadBlocks", numHistThreadBlocks, 0, sizeof(uint));
            cmd.DispatchCompute(m_CS, m_kernelGlobalHist, histIndirect, 0);

            cmd.SetComputeConstantBufferParam(m_CS, "cbGpuSortingNumThreadBlocks", numSortThreadBlocks, 0, sizeof(uint));
            cmd.DispatchCompute(m_CS, m_kernelScan, k_radixPasses, 1, 1);
            
            bool flip = false;
            for (int radixShift = 0; radixShift < bits; radixShift += 8)
            {
                cmd.SetComputeIntParam(m_CS, "e_radixShift", radixShift);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_sort", toSort);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_sortPayload", toSortPayload);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_alt", alt);
                cmd.SetComputeBufferParam(m_CS, m_digitBinningPass, "b_altPayload", altPayload);
                cmd.DispatchCompute(m_CS, m_digitBinningPass, sortIndirect, 0);

                (toSort, alt) = (alt, toSort);
                (toSortPayload, altPayload) = (altPayload, toSortPayload);
                flip = !flip;
            }

            if (flip && copyIfFlip) {
                // Copy _alt to _toSort
                cmd.SetComputeBufferParam(m_CS, m_kernelCopy, "b_sort", toSort);
                cmd.SetComputeBufferParam(m_CS, m_kernelCopy, "b_sortPayload", toSortPayload);
                cmd.SetComputeBufferParam(m_CS, m_kernelCopy, "b_alt", alt);
                cmd.SetComputeBufferParam(m_CS, m_kernelCopy, "b_altPayload", altPayload);
                cmd.DispatchCompute(m_CS, m_kernelCopy, copyIndirect, 0);
            }
        }

        public void Dispatch(CommandBuffer cmd, Args args)
        {
            Assert.IsTrue(Valid);
            cmd.DisableKeyword(m_CS, m_indirectKeyword);

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

        public void BeforeDispatchIndirect(CommandBuffer cmd, IndirectArgs args)
        {
            cmd.EnableKeyword(m_CS, m_indirectKeyword);
            cmd.SetComputeConstantBufferParam(m_CS, "cbGpuSortingNumKeys", args.count, 0, sizeof(uint));
            cmd.SetComputeBufferParam(m_CS, m_kernelInitIndirect, "b_histIndirect", args.resources.histIndirectBuffer);
            cmd.SetComputeBufferParam(m_CS, m_kernelInitIndirect, "b_sortIndirect", args.resources.sortIndirectBuffer);
            cmd.SetComputeBufferParam(m_CS, m_kernelInitIndirect, "b_copyIndirect", args.resources.copyIndirectBuffer);
            cmd.DispatchCompute(m_CS, m_kernelInitIndirect, 1, 1, 1);
        }

        public void DispatchIndirect(CommandBuffer cmd, IndirectArgs args, int bits = 32, bool copyIfFlip = true)
        {
            cmd.EnableKeyword(m_CS, m_indirectKeyword);
            Assert.IsTrue(Valid);

            SetStaticRootParameters(
                -1,
                cmd,
                args.inputKeys,
                args.resources.tempPassHistBuffer,
                args.resources.tempGlobalHistBuffer,
                args.resources.tempIndexBuffer);

            DispatchIndirect(
                args.resources.histIndirectBuffer,
                args.resources.sortIndirectBuffer,
                args.resources.copyIndirectBuffer,
                args.resources.histIndirectBuffer,
                args.resources.sortIndirectBuffer,
                cmd,
                args.inputKeys,
                args.inputValues,
                args.resources.tempKeyBuffer,
                args.resources.tempPayloadBuffer,
                bits,
                copyIfFlip);
        }
    }
}
