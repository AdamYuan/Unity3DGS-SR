// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace GaussianSplatting.Runtime
{
    public enum GaussianSplatRenderMode
    {
        Quad,
        Tile,
    }

    class GaussianSplatRenderSystem
    {
        // ReSharper disable MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        internal static readonly ProfilerMarker s_ProfDraw = new(ProfilerCategory.Render, "GaussianSplat.Draw", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCompose = new(ProfilerCategory.Render, "GaussianSplat.Compose", MarkerFlags.SampleGPU);
        internal static readonly ProfilerMarker s_ProfCalcView = new(ProfilerCategory.Render, "GaussianSplat.CalcView", MarkerFlags.SampleGPU);
        // ReSharper restore MemberCanBePrivate.Global

        public static GaussianSplatRenderSystem instance => ms_Instance ??= new GaussianSplatRenderSystem();
        static GaussianSplatRenderSystem ms_Instance;

        readonly Dictionary<GaussianSplatRenderer, MaterialPropertyBlock> m_Splats = new();
        readonly HashSet<Camera> m_CameraCommandBuffersDone = new();
        readonly List<(GaussianSplatRenderer, MaterialPropertyBlock)> m_ActiveSplats = new();

        CommandBuffer m_CommandBuffer;

        public void RegisterSplat(GaussianSplatRenderer r)
        {
            if (m_Splats.Count == 0)
            {
                if (GraphicsSettings.currentRenderPipeline == null)
                    Camera.onPreCull += OnPreCullCamera;
            }

            m_Splats.Add(r, new MaterialPropertyBlock());
        }

        public void UnregisterSplat(GaussianSplatRenderer r)
        {
            if (!m_Splats.ContainsKey(r))
                return;
            m_Splats.Remove(r);
            if (m_Splats.Count == 0)
            {
                if (m_CameraCommandBuffersDone != null)
                {
                    if (m_CommandBuffer != null)
                    {
                        foreach (var cam in m_CameraCommandBuffersDone)
                        {
                            if (cam)
                                cam.RemoveCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                        }
                    }
                    m_CameraCommandBuffersDone.Clear();
                }

                m_ActiveSplats.Clear();
                m_CommandBuffer?.Dispose();
                m_CommandBuffer = null;
                Camera.onPreCull -= OnPreCullCamera;
            }
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public bool GatherSplatsForCamera(Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return false;
            // gather all active & valid splat objects
            m_ActiveSplats.Clear();
            foreach (var kvp in m_Splats)
            {
                var gs = kvp.Key;
                if (gs == null || !gs.isActiveAndEnabled || !gs.HasValidAsset || !gs.HasValidRenderSetup)
                    continue;
                m_ActiveSplats.Add((kvp.Key, kvp.Value));
            }
            if (m_ActiveSplats.Count == 0)
                return false;

            // sort them by depth from camera
            var camTr = cam.transform;
            m_ActiveSplats.Sort((a, b) =>
            {
                var trA = a.Item1.transform;
                var trB = b.Item1.transform;
                var posA = camTr.InverseTransformPoint(trA.position);
                var posB = camTr.InverseTransformPoint(trB.position);
                return posA.z.CompareTo(posB.z);
            });

            return true;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public Material SortAndRenderSplats(Camera cam, CommandBuffer cmb)
        {
            Material matComposite = null;
            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;
                matComposite = gs.m_MatComposite;
                var mpb = kvp.Item2;
                var matrix = gs.transform.localToWorldMatrix;

                // cache view
                kvp.Item2.Clear();
                Material displayMat = gs.m_RenderMode switch
                {
                    GaussianSplatRenderer.RenderMode.DebugPoints => gs.m_MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugPointIndices => gs.m_MatDebugPoints,
                    GaussianSplatRenderer.RenderMode.DebugBoxes => gs.m_MatDebugBoxes,
                    GaussianSplatRenderer.RenderMode.DebugChunkBounds => gs.m_MatDebugBoxes,
                    _ => gs.m_MatSplats
                };
                if (displayMat == null)
                    continue;

                gs.SetAssetDataOnMaterial(mpb);
                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatChunks, gs.m_GpuChunks);

                mpb.SetBuffer(GaussianSplatRenderer.Props.SplatViewData, gs.m_GpuView);

                mpb.SetBuffer(GaussianSplatRenderer.Props.OrderBuffer, gs.m_GpuSortKeys);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatScale, gs.m_SplatScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatOpacityScale, gs.m_OpacityScale);
                mpb.SetFloat(GaussianSplatRenderer.Props.SplatSize, gs.m_PointDisplaySize);
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOrder, gs.m_SHOrder);
                mpb.SetInteger(GaussianSplatRenderer.Props.SHOnly, gs.m_SHOnly ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayIndex, gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugPointIndices ? 1 : 0);
                mpb.SetInteger(GaussianSplatRenderer.Props.DisplayChunks, gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugChunkBounds ? 1 : 0);

                cmb.BeginSample(s_ProfCalcView);
                gs.CalcViewData(cmb, cam, matrix);
                cmb.EndSample(s_ProfCalcView);

                // sort
                gs.SortPoints(cmb, cam, matrix);
                ++gs.m_FrameCounter;

                // draw
                int indexCount = 6;
                int instanceCount = gs.splatCount;
                MeshTopology topology = MeshTopology.Triangles;
                if (gs.m_RenderMode is GaussianSplatRenderer.RenderMode.DebugBoxes or GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    indexCount = 36;
                if (gs.m_RenderMode == GaussianSplatRenderer.RenderMode.DebugChunkBounds)
                    instanceCount = gs.m_GpuChunksValid ? gs.m_GpuChunks.count : 0;

                cmb.BeginSample(s_ProfDraw);
                if (gs.m_RenderMode == GaussianSplatRenderer.RenderMode.Splats)
                    cmb.DrawProceduralIndirect(gs.m_GpuIndexBuffer, matrix, displayMat, 0, topology, gs.m_GpuDrawIndirectBuffer, 0, mpb);
                else
                    cmb.DrawProcedural(gs.m_GpuIndexBuffer, matrix, displayMat, 0, topology, indexCount, instanceCount, mpb);
                cmb.EndSample(s_ProfDraw);
            }
            return matComposite;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        public Material TileSortAndRenderSplats(Camera cam, CommandBuffer cmb, RTHandle target)
        {
            Material matComposite = null;
            foreach (var kvp in m_ActiveSplats)
            {
                var gs = kvp.Item1;
                matComposite = gs.m_MatComposite;
                var mpb = kvp.Item2;
                var matrix = gs.transform.localToWorldMatrix;

                // cache view
                kvp.Item2.Clear();

                cmb.BeginSample(s_ProfCalcView);
                gs.CalcTileViewData(cmb, cam);
                cmb.EndSample(s_ProfCalcView);

                gs.SortTilePoints(cmb, cam);
                gs.CalcTileRanges(cmb, cam);

                gs.RenderTiles(cmb, cam, target);
            }
            return matComposite;
        }

        // ReSharper disable once MemberCanBePrivate.Global - used by HDRP/URP features that are not always compiled
        // ReSharper disable once UnusedMethodReturnValue.Global - used by HDRP/URP features that are not always compiled
        public CommandBuffer InitialClearCmdBuffer(Camera cam)
        {
            m_CommandBuffer ??= new CommandBuffer { name = "RenderGaussianSplats" };
            if (GraphicsSettings.currentRenderPipeline == null && cam != null && !m_CameraCommandBuffersDone.Contains(cam))
            {
                cam.AddCommandBuffer(CameraEvent.BeforeForwardAlpha, m_CommandBuffer);
                m_CameraCommandBuffersDone.Add(cam);
            }

            // get render target for all splats
            m_CommandBuffer.Clear();
            return m_CommandBuffer;
        }

        void OnPreCullCamera(Camera cam)
        {
            if (!GatherSplatsForCamera(cam))
                return;

            InitialClearCmdBuffer(cam);

            m_CommandBuffer.GetTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT, -1, -1, 0, FilterMode.Point, GraphicsFormat.R16G16B16A16_SFloat);
            m_CommandBuffer.SetRenderTarget(GaussianSplatRenderer.Props.GaussianSplatRT, BuiltinRenderTextureType.CurrentActive);
            m_CommandBuffer.ClearRenderTarget(RTClearFlags.Color, new Color(0, 0, 0, 0), 0, 0);

            // add sorting, view calc and drawing commands for each splat object
            Material matComposite = SortAndRenderSplats(cam, m_CommandBuffer);

            // compose
            m_CommandBuffer.BeginSample(s_ProfCompose);
            m_CommandBuffer.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
            m_CommandBuffer.DrawProcedural(Matrix4x4.identity, matComposite, 0, MeshTopology.Triangles, 3, 1);
            m_CommandBuffer.EndSample(s_ProfCompose);
            m_CommandBuffer.ReleaseTemporaryRT(GaussianSplatRenderer.Props.GaussianSplatRT);
        }
    }

    [ExecuteInEditMode]
    public class GaussianSplatRenderer : MonoBehaviour
    {
        public enum RenderMode
        {
            Splats,
            DebugPoints,
            DebugPointIndices,
            DebugBoxes,
            DebugChunkBounds,
        }

        public enum SplatRenderMode
        {
            Quad,
            Tile
        }

        public const int kLogSplatTileSize = 4; // For tile-based renderer
        public const int kSplatTileSize = 1 << kLogSplatTileSize; // For tile-based renderer
        public const int kSplatTileBufferRatio = 3; // For tile-based renderer
        public const int kMaxTilesOnScreen = 1920 * 1080 * 4 / kSplatTileSize / kSplatTileSize; // Max possible tiles on screen

        // Constants For Splat splitting
        public const int kGpuSubSplatDataSize = 32; // Size of struct SubSplatData
        public const int kLogMaxSubSplatCount = 18; // Don't change this
        public const int kMaxSubSplatCount = 1 << kLogMaxSubSplatCount; // Don't change this

        public GaussianSplatAsset m_Asset;

        [Range(0.1f, 2.0f)]
        [Tooltip("Additional scaling factor for the splats")]
        public float m_SplatScale = 1.0f;
        [Range(0.05f, 20.0f)]
        [Tooltip("Additional scaling factor for opacity")]
        public float m_OpacityScale = 1.0f;
        [Range(0, 3)]
        [Tooltip("Spherical Harmonics order to use")]
        public int m_SHOrder = 3;
        [Tooltip("Show only Spherical Harmonics contribution, using gray color")]
        public bool m_SHOnly;
        
        public bool m_EnableSubSplats = true;
        public bool m_LockSubSplats = false;
        public bool m_StepSubSplats = false;

        public RenderMode m_RenderMode = RenderMode.Splats;
        [Range(1.0f, 15.0f)] public float m_PointDisplaySize = 3.0f;

        public GaussianCutout[] m_Cutouts;

        public Shader m_ShaderSplats;
        public Shader m_ShaderComposite;
        public Shader m_ShaderDebugPoints;
        public Shader m_ShaderDebugBoxes;
        [Tooltip("Gaussian splatting compute shader")]
        public ComputeShader m_CSSplatUtilities;

        int m_SplatCount; // initially same as asset splat count, but editing can change this

        // Sort pairs for Quad Renderer
        GraphicsBuffer m_GpuSortDistances;
        internal GraphicsBuffer m_GpuSortKeys;
        GraphicsBuffer m_GpuViewSplatCount;

        // Sort pairs for Tile-based Renderer
        GraphicsBuffer m_GpuTileSortTileDist;
        GraphicsBuffer m_GpuTileSortKeys;
        // Other data for Tile-based Renderer
        GraphicsBuffer m_GpuTileSplatCount;
        GraphicsBuffer m_GpuTileRanges;
        uint2[] m_GpuTileRangesClearData;
        GraphicsBuffer m_GpuTileSplatIndirect;
        
        // Buffers for Sub-Splat
        GraphicsBuffer m_GpuSubSplatIDStack;
        GraphicsBuffer m_GpuSubSplatCount;
        GraphicsBuffer m_GpuSubSplatParents;
        GraphicsBuffer m_GpuSubSplatRoots;
        GraphicsBuffer m_GpuSubSplats;
        GraphicsBuffer m_GpuSubSplatRefs0;
        GraphicsBuffer m_GpuSubSplatRefs1;
        GraphicsBuffer m_GpuSubSplatSplitRefs;
        GraphicsBuffer m_GpuSubSplatMergeRefs;
        GraphicsBuffer m_GpuSubSplatInitIDs;
        GraphicsBuffer m_GpuSubSplatRefCount;
        GraphicsBuffer m_GpuSubSplatRefCountConst;
        GraphicsBuffer m_GpuSubSplatRefIndirect;
        GraphicsBuffer m_GpuSubSplatLevels;
        bool m_SubSplatRefFlip = false;


        GraphicsBuffer m_GpuPosData;
        GraphicsBuffer m_GpuOtherData;
        GraphicsBuffer m_GpuSHData;
        Texture m_GpuColorData;
        internal GraphicsBuffer m_GpuChunks;
        internal bool m_GpuChunksValid;
        internal GraphicsBuffer m_GpuView;
        internal GraphicsBuffer m_GpuDrawIndirectBuffer;
        internal GraphicsBuffer m_GpuIndexBuffer;

        // these buffers are only for splat editing, and are lazily created
        GraphicsBuffer m_GpuEditCutouts;
        GraphicsBuffer m_GpuEditCountsBounds;
        GraphicsBuffer m_GpuEditSelected;
        GraphicsBuffer m_GpuEditDeleted;
        GraphicsBuffer m_GpuEditSelectedMouseDown; // selection state at start of operation
        GraphicsBuffer m_GpuEditPosMouseDown; // position state at start of operation
        GraphicsBuffer m_GpuEditOtherMouseDown; // rotation/scale state at start of operation

        GpuSorting m_Sorter;
        GpuSorting.IndirectArgs m_SorterArgs;
        GpuSorting.IndirectArgs m_TileSorterArgs;

        internal Material m_MatSplats;
        internal Material m_MatComposite;
        internal Material m_MatDebugPoints;
        internal Material m_MatDebugBoxes;

        internal int m_FrameCounter;
        GaussianSplatAsset m_PrevAsset;
        Hash128 m_PrevHash;


        static readonly ProfilerMarker s_ProfSort = new(ProfilerCategory.Render, "GaussianSplat.Sort", MarkerFlags.SampleGPU);

        internal static class Props
        {
            public static readonly int SplatPos = Shader.PropertyToID("_SplatPos");
            public static readonly int SplatOther = Shader.PropertyToID("_SplatOther");
            public static readonly int SplatSH = Shader.PropertyToID("_SplatSH");
            public static readonly int SplatColor = Shader.PropertyToID("_SplatColor");
            public static readonly int SplatSelectedBits = Shader.PropertyToID("_SplatSelectedBits");
            public static readonly int SplatDeletedBits = Shader.PropertyToID("_SplatDeletedBits");
            public static readonly int SplatBitsValid = Shader.PropertyToID("_SplatBitsValid");
            public static readonly int SplatFormat = Shader.PropertyToID("_SplatFormat");
            public static readonly int SplatChunks = Shader.PropertyToID("_SplatChunks");
            public static readonly int SplatChunkCount = Shader.PropertyToID("_SplatChunkCount");
            public static readonly int SplatViewData = Shader.PropertyToID("_SplatViewData");
            public static readonly int OrderBuffer = Shader.PropertyToID("_OrderBuffer");
            public static readonly int SplatScale = Shader.PropertyToID("_SplatScale");
            public static readonly int SplatOpacityScale = Shader.PropertyToID("_SplatOpacityScale");
            public static readonly int SplatSize = Shader.PropertyToID("_SplatSize");
            public static readonly int SplatCount = Shader.PropertyToID("_SplatCount");
            public static readonly int SHOrder = Shader.PropertyToID("_SHOrder");
            public static readonly int SHOnly = Shader.PropertyToID("_SHOnly");
            public static readonly int DisplayIndex = Shader.PropertyToID("_DisplayIndex");
            public static readonly int DisplayChunks = Shader.PropertyToID("_DisplayChunks");
            public static readonly int GaussianSplatRT = Shader.PropertyToID("_GaussianSplatRT");
            public static readonly int SplatSortKeys = Shader.PropertyToID("_SplatSortKeys");
            public static readonly int SplatSortDistances = Shader.PropertyToID("_SplatSortDistances");
            public static readonly int SplatTileViewData = Shader.PropertyToID("_SplatTileViewData");
            public static readonly int SplatTileViewDataRO = Shader.PropertyToID("_SplatTileViewDataRO");
            public static readonly int ViewSplatCount = Shader.PropertyToID("_ViewSplatCount");
            public static readonly int ViewSplatDrawIndirect = Shader.PropertyToID("_ViewSplatDrawIndirect");
            public static readonly int TileSplatSortKeys = Shader.PropertyToID("_TileSplatSortKeys");
            public static readonly int TileSplatSortKeysRO = Shader.PropertyToID("_TileSplatSortKeysRO");
            public static readonly int TileSplatSortDistances = Shader.PropertyToID("_TileSplatSortDistances");
            public static readonly int TileSplatSortTiles = Shader.PropertyToID("_TileSplatSortTiles");
            public static readonly int TileSplatSortTilesRO = Shader.PropertyToID("_TileSplatSortTilesRO");
            public static readonly int TileSplatCount = Shader.PropertyToID("_TileSplatCount");
            public static readonly int TileSplatIndirect = Shader.PropertyToID("_TileSplatIndirect");
            public static readonly int TileRanges = Shader.PropertyToID("_TileRanges");
            public static readonly int TileRangesRO = Shader.PropertyToID("_TileRangesRO");
            public static readonly int CBTileSplatCount = Shader.PropertyToID("cbTileSplatCount");
            public static readonly int RenderTarget = Shader.PropertyToID("_RenderTarget");
            
            // BEGIN Props for Splat Splitting
            public static readonly int SubSplatLocked = Shader.PropertyToID("_SubSplatLocked");
            public static readonly int SubSplatIDStack = Shader.PropertyToID("_SubSplatIDStack");
            public static readonly int SubSplatCount = Shader.PropertyToID("_SubSplatCount");
            public static readonly int SubSplatParents = Shader.PropertyToID("_SubSplatParents");
            public static readonly int SubSplatRoots = Shader.PropertyToID("_SubSplatRoots");
            public static readonly int SubSplats = Shader.PropertyToID("_SubSplats");
            public static readonly int SubSplatSrcRefs = Shader.PropertyToID("_SubSplatSrcRefs");
            public static readonly int SubSplatDstRefs = Shader.PropertyToID("_SubSplatDstRefs");
            public static readonly int SubSplatSplitRefs = Shader.PropertyToID("_SubSplatSplitRefs");
            public static readonly int SubSplatMergeRefs = Shader.PropertyToID("_SubSplatMergeRefs");
            public static readonly int SubSplatInitIDs = Shader.PropertyToID("_SubSplatInitIDs");
            public static readonly int SubSplatRefCount = Shader.PropertyToID("_SubSplatRefCount");
            public static readonly int CBSubSplatRefCount = Shader.PropertyToID("cbSubSplatRefCount");
            public static readonly int SubSplatRefIndirect = Shader.PropertyToID("_SubSplatRefIndirect");
            public static readonly int SubSplatLevels = Shader.PropertyToID("_SubSplatLevels");
            // END Props for Splat Splitting

            public static readonly int SrcBuffer = Shader.PropertyToID("_SrcBuffer");
            public static readonly int DstBuffer = Shader.PropertyToID("_DstBuffer");
            public static readonly int BufferSize = Shader.PropertyToID("_BufferSize");
            public static readonly int MatrixVP = Shader.PropertyToID("_MatrixVP");
            public static readonly int MatrixMV = Shader.PropertyToID("_MatrixMV");
            public static readonly int MatrixP = Shader.PropertyToID("_MatrixP");
            public static readonly int MatrixObjectToWorld = Shader.PropertyToID("_MatrixObjectToWorld");
            public static readonly int MatrixWorldToObject = Shader.PropertyToID("_MatrixWorldToObject");
            public static readonly int VecScreenParams = Shader.PropertyToID("_VecScreenParams");
            public static readonly int VecTileParams = Shader.PropertyToID("_VecTileParams");
            public static readonly int VecWorldSpaceCameraPos = Shader.PropertyToID("_VecWorldSpaceCameraPos");
            public static readonly int SelectionCenter = Shader.PropertyToID("_SelectionCenter");
            public static readonly int SelectionDelta = Shader.PropertyToID("_SelectionDelta");
            public static readonly int SelectionDeltaRot = Shader.PropertyToID("_SelectionDeltaRot");
            public static readonly int SplatCutoutsCount = Shader.PropertyToID("_SplatCutoutsCount");
            public static readonly int SplatCutouts = Shader.PropertyToID("_SplatCutouts");
            public static readonly int SelectionMode = Shader.PropertyToID("_SelectionMode");
            public static readonly int SplatPosMouseDown = Shader.PropertyToID("_SplatPosMouseDown");
            public static readonly int SplatOtherMouseDown = Shader.PropertyToID("_SplatOtherMouseDown");
        }

        [field: NonSerialized] public bool editModified { get; private set; }
        [field: NonSerialized] public uint editSelectedSplats { get; private set; }
        [field: NonSerialized] public uint editDeletedSplats { get; private set; }
        [field: NonSerialized] public uint editCutSplats { get; private set; }
        [field: NonSerialized] public Bounds editSelectedBounds { get; private set; }

        public GaussianSplatAsset asset => m_Asset;
        public int splatCount => m_SplatCount;

        enum KernelIndices
        {
            InitSubSplatIndirect,
            SplitSubSplats,
            MergeSubSplats,
            InitSubSplats,
            CalcViewData,
            CalcRootViewData,
            CalcSubViewData,
            InitViewSplatIndirect,
            CalcTileViewData,
            InitTileSplatIndirect,
            ReorderTileID,
            CalcTileRanges,
            RenderTile,
            UpdateEditData,
            InitEditData,
            ClearBuffer,
            InvertSelection,
            SelectAll,
            OrBuffers,
            SelectionUpdate,
            TranslateSelection,
            RotateSelection,
            ScaleSelection,
            ExportData,
            CopySplats,
        }

        public bool HasValidAsset =>
            m_Asset != null &&
            m_Asset.splatCount > 0 &&
            m_Asset.formatVersion == GaussianSplatAsset.kCurrentVersion &&
            m_Asset.posData != null &&
            m_Asset.otherData != null &&
            m_Asset.shData != null &&
            m_Asset.colorData != null;
        public bool HasValidRenderSetup => m_GpuPosData != null && m_GpuOtherData != null && m_GpuChunks != null;

        const int kGpuViewDataSize = 24;

        void CreateResourcesForAsset()
        {
            if (!HasValidAsset)
                return;

            m_SplatCount = asset.splatCount;
            m_GpuPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int)(asset.posData.dataSize / 4), 4) { name = "GaussianPosData" };
            m_GpuPosData.SetData(asset.posData.GetData<uint>());
            m_GpuOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, (int)(asset.otherData.dataSize / 4), 4) { name = "GaussianOtherData" };
            m_GpuOtherData.SetData(asset.otherData.GetData<uint>());
            m_GpuSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, (int)(asset.shData.dataSize / 4), 4) { name = "GaussianSHData" };
            m_GpuSHData.SetData(asset.shData.GetData<uint>());
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(asset.splatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            var tex = new Texture2D(texWidth, texHeight, texFormat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.IgnoreMipmapLimit | TextureCreationFlags.DontUploadUponCreate) { name = "GaussianColorData" };
            tex.SetPixelData(asset.colorData.GetData<byte>(), 0);
            tex.Apply(false, true);
            m_GpuColorData = tex;
            if (asset.chunkData != null && asset.chunkData.dataSize != 0)
            {
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    (int)(asset.chunkData.dataSize / UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()),
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>())
                { name = "GaussianChunkData" };
                m_GpuChunks.SetData(asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
                m_GpuChunksValid = true;
            }
            else
            {
                // just a dummy chunk buffer
                m_GpuChunks = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1,
                    UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>())
                { name = "GaussianChunkData" };
                m_GpuChunksValid = false;
            }

            m_GpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, m_Asset.splatCount, kGpuViewDataSize);
            m_GpuIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Index, 36, 2);
            // cube indices, most often we use only the first quad
            m_GpuIndexBuffer.SetData(new ushort[]
            {
                0, 1, 2, 1, 3, 2,
                4, 6, 5, 5, 6, 7,
                0, 2, 4, 4, 2, 6,
                1, 5, 3, 5, 7, 3,
                0, 4, 1, 4, 5, 1,
                2, 3, 6, 3, 7, 6
            });

            m_GpuViewSplatCount = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Constant, 1, sizeof(uint));
            m_GpuDrawIndirectBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments, 5, sizeof(uint));

            m_GpuTileSplatCount = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Constant, 1, sizeof(uint));
            m_GpuTileSplatIndirect = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments, 3, sizeof(uint));
            m_GpuTileRanges = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxTilesOnScreen, 2 * sizeof(uint)) { name = "TileRanges" }; ;
            m_GpuTileRangesClearData = new uint2[kMaxTilesOnScreen];

            InitSortBuffers(splatCount);
            InitSubSplatBuffers(splatCount);
        }
        
        void InitSubSplatBuffers(int count) {
            m_GpuSubSplatIDStack = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            uint[] splatIDStack = new uint[kMaxSubSplatCount / 2];
            for (uint i = 0; i < kMaxSubSplatCount / 2; ++i)
                splatIDStack[i] = i * 2;
            m_GpuSubSplatIDStack.SetData(splatIDStack);
            
            m_GpuSubSplatCount = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
            m_GpuSubSplatCount.SetData(new uint[1]);

            m_GpuSubSplatParents = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            m_GpuSubSplatRoots = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            m_GpuSubSplats = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount, kGpuSubSplatDataSize);

            m_GpuSubSplatRefs0 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            m_GpuSubSplatRefs1 = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            m_GpuSubSplatSplitRefs = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            m_GpuSubSplatMergeRefs = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            m_GpuSubSplatInitIDs = new GraphicsBuffer(GraphicsBuffer.Target.Structured, kMaxSubSplatCount / 2, sizeof(uint));
            m_GpuSubSplatRefCount = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.CopySource, 
                10, sizeof(uint)
            );
            m_GpuSubSplatRefCount.SetData(new uint[10]);
            m_GpuSubSplatRefCountConst = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.CopyDestination | GraphicsBuffer.Target.Constant, 
                10, sizeof(uint)
            );
            m_GpuSubSplatRefCountConst.SetData(new uint[10]);
            m_GpuSubSplatRefIndirect = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments, 
                12, sizeof(uint)
            );
            m_GpuSubSplatRefIndirect.SetData(new uint[12]{0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1});

            m_GpuSubSplatLevels = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (count + 7) / 8, sizeof(uint));
            m_GpuSubSplatLevels.SetData(new uint[(count + 7) / 8]);
        }

        void InitSortBuffers(int count)
        {
            m_GpuSortDistances?.Dispose();
            m_GpuSortKeys?.Dispose();
            m_SorterArgs.resources.Dispose();

            DisposeBuffer(ref m_GpuTileSortTileDist);
            DisposeBuffer(ref m_GpuTileSortKeys);
            m_TileSorterArgs.resources.Dispose();

            m_GpuSortDistances = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortDistances" };
            m_GpuSortKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 4) { name = "GaussianSplatSortIndices" };

            m_SorterArgs.inputKeys = m_GpuSortDistances;
            m_SorterArgs.inputValues = m_GpuSortKeys;
            m_SorterArgs.count = m_GpuViewSplatCount;
            if (m_Sorter.Valid)
                m_SorterArgs.resources = GpuSorting.SupportResources.LoadIndirect((uint)count); // TODO: Might need a ratio

            m_GpuTileSortTileDist = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count * kSplatTileBufferRatio, 4) { name = "GaussianSplatTileSortTileDist" };
            m_GpuTileSortKeys = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count * kSplatTileBufferRatio, 4) { name = "GaussianSplatTileSortIndices" };

            m_TileSorterArgs.inputKeys = m_GpuTileSortTileDist;
            m_TileSorterArgs.inputValues = m_GpuTileSortKeys;
            m_TileSorterArgs.count = m_GpuTileSplatCount;
            if (m_Sorter.Valid)
                m_TileSorterArgs.resources = GpuSorting.SupportResources.LoadIndirect((uint)count * kSplatTileBufferRatio);
        }

        public void OnEnable()
        {
            m_FrameCounter = 0;
            if (m_ShaderSplats == null || m_ShaderComposite == null || m_ShaderDebugPoints == null || m_ShaderDebugBoxes == null || m_CSSplatUtilities == null)
                return;
            if (!SystemInfo.supportsComputeShaders)
                return;

            m_MatSplats = new Material(m_ShaderSplats) { name = "GaussianSplats" };
            m_MatComposite = new Material(m_ShaderComposite) { name = "GaussianClearDstAlpha" };
            m_MatDebugPoints = new Material(m_ShaderDebugPoints) { name = "GaussianDebugPoints" };
            m_MatDebugBoxes = new Material(m_ShaderDebugBoxes) { name = "GaussianDebugBoxes" };

            m_Sorter = new GpuSorting(m_CSSplatUtilities);
            GaussianSplatRenderSystem.instance.RegisterSplat(this);

            CreateResourcesForAsset();
        }

        void SetAssetDataOnCS(CommandBuffer cmb, KernelIndices kernel)
        {
            ComputeShader cs = m_CSSplatUtilities;
            int kernelIndex = (int)kernel;
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatPos, m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatChunks, m_GpuChunks);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatOther, m_GpuOtherData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSH, m_GpuSHData);
            cmb.SetComputeTextureParam(cs, kernelIndex, Props.SplatColor, m_GpuColorData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSelectedBits, m_GpuEditSelected ?? m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatDeletedBits, m_GpuEditDeleted ?? m_GpuPosData);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatViewData, m_GpuView);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.OrderBuffer, m_GpuSortKeys);

            cmb.SetComputeIntParam(cs, Props.SplatBitsValid, m_GpuEditSelected != null && m_GpuEditDeleted != null ? 1 : 0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            cmb.SetComputeIntParam(cs, Props.SplatFormat, (int)format);
            cmb.SetComputeIntParam(cs, Props.SplatCount, m_SplatCount);
            cmb.SetComputeIntParam(cs, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);

            UpdateCutoutsBuffer();
            cmb.SetComputeIntParam(cs, Props.SplatCutoutsCount, m_Cutouts?.Length ?? 0);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatCutouts, m_GpuEditCutouts);
        }

        internal void SetAssetDataOnMaterial(MaterialPropertyBlock mat)
        {
            mat.SetBuffer(Props.SplatPos, m_GpuPosData);
            mat.SetBuffer(Props.SplatOther, m_GpuOtherData);
            mat.SetBuffer(Props.SplatSH, m_GpuSHData);
            mat.SetTexture(Props.SplatColor, m_GpuColorData);
            mat.SetBuffer(Props.SplatSelectedBits, m_GpuEditSelected ?? m_GpuPosData);
            mat.SetBuffer(Props.SplatDeletedBits, m_GpuEditDeleted ?? m_GpuPosData);
            mat.SetInt(Props.SplatBitsValid, m_GpuEditSelected != null && m_GpuEditDeleted != null ? 1 : 0);
            uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
            mat.SetInteger(Props.SplatFormat, (int)format);
            mat.SetInteger(Props.SplatCount, m_SplatCount);
            mat.SetInteger(Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);
        }

        void SetSubSplatDataOnCS(CommandBuffer cmb, KernelIndices kernel)
        {
            ComputeShader cs = m_CSSplatUtilities;
            int kernelIndex = (int)kernel;

            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatIDStack, m_GpuSubSplatIDStack);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatCount, m_GpuSubSplatCount);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatParents, m_GpuSubSplatParents);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatRoots, m_GpuSubSplatRoots);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplats, m_GpuSubSplats);

            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatSrcRefs, m_SubSplatRefFlip ? m_GpuSubSplatRefs0 : m_GpuSubSplatRefs1);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatDstRefs, m_SubSplatRefFlip ? m_GpuSubSplatRefs1 : m_GpuSubSplatRefs0);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatSplitRefs, m_GpuSubSplatSplitRefs);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatMergeRefs, m_GpuSubSplatMergeRefs);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatInitIDs, m_GpuSubSplatInitIDs);
            cmb.SetComputeConstantBufferParam(cs, Props.CBSubSplatRefCount, m_GpuSubSplatRefCountConst, 0, 10 * sizeof(uint));
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatRefCount, m_GpuSubSplatRefCount);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatRefIndirect, m_GpuSubSplatRefIndirect);
            cmb.SetComputeBufferParam(cs, kernelIndex, Props.SubSplatLevels, m_GpuSubSplatLevels);
        }

        static void DisposeBuffer(ref GraphicsBuffer buf)
        {
            buf?.Dispose();
            buf = null;
        }

        void DisposeResourcesForAsset()
        {
            DestroyImmediate(m_GpuColorData);

            DisposeBuffer(ref m_GpuPosData);
            DisposeBuffer(ref m_GpuOtherData);
            DisposeBuffer(ref m_GpuSHData);
            DisposeBuffer(ref m_GpuChunks);

            DisposeBuffer(ref m_GpuView);
            DisposeBuffer(ref m_GpuIndexBuffer);
            DisposeBuffer(ref m_GpuSortDistances);
            DisposeBuffer(ref m_GpuSortKeys);

            DisposeBuffer(ref m_GpuViewSplatCount);
            DisposeBuffer(ref m_GpuDrawIndirectBuffer);

            DisposeBuffer(ref m_GpuTileSortTileDist);
            DisposeBuffer(ref m_GpuTileSortKeys);
            DisposeBuffer(ref m_GpuTileSplatCount);
            DisposeBuffer(ref m_GpuTileRanges);
            DisposeBuffer(ref m_GpuTileSplatIndirect);
            
            DisposeBuffer(ref m_GpuSubSplatIDStack);
            DisposeBuffer(ref m_GpuSubSplatCount);
            DisposeBuffer(ref m_GpuSubSplatParents);
            DisposeBuffer(ref m_GpuSubSplatRoots);
            DisposeBuffer(ref m_GpuSubSplats);
            DisposeBuffer(ref m_GpuSubSplatRefs0);
            DisposeBuffer(ref m_GpuSubSplatRefs1);
            DisposeBuffer(ref m_GpuSubSplatSplitRefs);
            DisposeBuffer(ref m_GpuSubSplatMergeRefs);
            DisposeBuffer(ref m_GpuSubSplatInitIDs);
            DisposeBuffer(ref m_GpuSubSplatRefCount);
            DisposeBuffer(ref m_GpuSubSplatRefCountConst);
            DisposeBuffer(ref m_GpuSubSplatRefIndirect);
            DisposeBuffer(ref m_GpuSubSplatLevels);

            DisposeBuffer(ref m_GpuEditSelectedMouseDown);
            DisposeBuffer(ref m_GpuEditPosMouseDown);
            DisposeBuffer(ref m_GpuEditOtherMouseDown);
            DisposeBuffer(ref m_GpuEditSelected);
            DisposeBuffer(ref m_GpuEditDeleted);
            DisposeBuffer(ref m_GpuEditCountsBounds);
            DisposeBuffer(ref m_GpuEditCutouts);

            m_SorterArgs.resources.Dispose();
            m_TileSorterArgs.resources.Dispose();

            m_SplatCount = 0;
            m_GpuChunksValid = false;

            editSelectedSplats = 0;
            editDeletedSplats = 0;
            editCutSplats = 0;
            editModified = false;
            editSelectedBounds = default;
        }

        public void OnDisable()
        {
            DisposeResourcesForAsset();
            GaussianSplatRenderSystem.instance.UnregisterSplat(this);

            DestroyImmediate(m_MatSplats);
            DestroyImmediate(m_MatComposite);
            DestroyImmediate(m_MatDebugPoints);
            DestroyImmediate(m_MatDebugBoxes);
        }

        internal void CalcViewData(CommandBuffer cmb, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            Vector4 screenPar = new(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixVP, matProj * matView);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixP, matProj);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatScale, m_SplatScale);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatOpacityScale, m_OpacityScale);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOrder, m_SHOrder);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOnly, m_SHOnly ? 1 : 0);

            // Debug 
            /*
            var subSplatCountData = new uint[1];
            var subSplatRefCountData = new uint[10];
            m_GpuSubSplatCount.GetData(subSplatCountData);
            m_GpuSubSplatRefCountConst.GetData(subSplatRefCountData);
            Debug.Log("cnt:" + subSplatCountData[0] * 2 + "; refCnt:" + subSplatRefCountData[0] + "; sRefCnt:" + subSplatRefCountData[1] + "; mRefCnt:" + subSplatRefCountData[2]);
            */

            // Reset View Counters
            cmb.SetBufferData(m_GpuViewSplatCount, new uint[1]);
            
            if (m_EnableSubSplats) 
            {
                bool notLocked = !m_LockSubSplats || m_StepSubSplats;
                m_StepSubSplats = false;

                cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SubSplatLocked, notLocked ? 0 : 1);

                // CalcRootViewData
                SetAssetDataOnCS(cmb, KernelIndices.CalcRootViewData);
                SetSubSplatDataOnCS(cmb, KernelIndices.CalcRootViewData);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcRootViewData, Props.SplatSortDistances, m_GpuSortDistances);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcRootViewData, Props.SplatSortKeys, m_GpuSortKeys);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcRootViewData, Props.ViewSplatCount, m_GpuViewSplatCount);
                m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcRootViewData, out uint gsX, out _, out _);
                cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcRootViewData, (m_GpuView.count + (int)gsX - 1) / (int)gsX, 1, 1);

                // CalcSubViewData
                SetAssetDataOnCS(cmb, KernelIndices.CalcSubViewData);
                SetSubSplatDataOnCS(cmb, KernelIndices.CalcSubViewData);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcSubViewData, Props.SplatSortDistances, m_GpuSortDistances);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcSubViewData, Props.SplatSortKeys, m_GpuSortKeys);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcSubViewData, Props.ViewSplatCount, m_GpuViewSplatCount);
                cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcSubViewData, m_GpuSubSplatRefIndirect, 0);
                
                if (notLocked) 
                {
                    // InitSubSplatIndirect
                    SetSubSplatDataOnCS(cmb, KernelIndices.InitSubSplatIndirect);
                    cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.InitSubSplatIndirect, 1, 1, 1);
                    cmb.CopyBuffer(m_GpuSubSplatRefCount, m_GpuSubSplatRefCountConst);

                    // Reset Sub-Splat Ref Counters
                    cmb.SetBufferData(m_GpuSubSplatRefCount, new uint[10]);
                 
                    // MergeSubSplats
                    SetSubSplatDataOnCS(cmb, KernelIndices.MergeSubSplats);
                    cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.MergeSubSplats, m_GpuSubSplatRefIndirect, 6 * sizeof(uint));
                 
                    // InitSubSplats
                    SetAssetDataOnCS(cmb, KernelIndices.InitSubSplats);
                    SetSubSplatDataOnCS(cmb, KernelIndices.InitSubSplats);
                    cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.InitSubSplats, m_GpuSubSplatRefIndirect, 9 * sizeof(uint));
                    
                    // SplitSubSplats
                    SetSubSplatDataOnCS(cmb, KernelIndices.SplitSubSplats);
                    cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.SplitSubSplats, m_GpuSubSplatRefIndirect, 3 * sizeof(uint));

                    m_SubSplatRefFlip = !m_SubSplatRefFlip;
                }
            } 
            else 
            {
                // CalcViewData
                SetAssetDataOnCS(cmb, KernelIndices.CalcViewData);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, Props.SplatSortDistances, m_GpuSortDistances);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, Props.SplatSortKeys, m_GpuSortKeys);
                cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, Props.ViewSplatCount, m_GpuViewSplatCount);
                m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcViewData, out uint gsX, out _, out _);
                cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcViewData, (m_GpuView.count + (int)gsX - 1) / (int)gsX, 1, 1);
            }

            // InitViewSplatIndirect
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitViewSplatIndirect, Props.ViewSplatCount, m_GpuViewSplatCount);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitViewSplatIndirect, Props.ViewSplatDrawIndirect, m_GpuDrawIndirectBuffer);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.InitViewSplatIndirect, 1, 1, 1);
        }

        internal void SortPoints(CommandBuffer cmd, Camera cam, Matrix4x4 matrix)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            // sort the splats
            cmd.BeginSample(s_ProfSort);
            m_Sorter.BeforeDispatchIndirect(cmd, m_SorterArgs);
            m_Sorter.DispatchIndirect(cmd, m_SorterArgs);
            cmd.EndSample(s_ProfSort);
        }

        internal void CalcTileViewData(CommandBuffer cmb, Camera cam)
        {
            Debug.Assert(m_SplatCount <= 0xFFFFFFu); // We need to pack 8-bit tile offset upon 24-bit splat ID

            if (cam.cameraType == CameraType.Preview)
                return;

            var tr = transform;

            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, false);
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int tileCountW = (screenW + kSplatTileSize - 1) / kSplatTileSize;
            int tileCountH = (screenH + kSplatTileSize - 1) / kSplatTileSize;
            Vector4 screenPar = new(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            // calculate view dependent data for each splat
            {
                ComputeShader cs = m_CSSplatUtilities;
                int kernelIndex = (int)KernelIndices.CalcTileViewData;
                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatPos, m_GpuPosData);
                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatChunks, m_GpuChunks);
                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatOther, m_GpuOtherData);
                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSH, m_GpuSHData);
                cmb.SetComputeTextureParam(cs, kernelIndex, Props.SplatColor, m_GpuColorData);
                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatSelectedBits, m_GpuEditSelected ?? m_GpuPosData);
                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatDeletedBits, m_GpuEditDeleted ?? m_GpuPosData);

                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatTileViewData, m_GpuView); // The difference

                cmb.SetComputeIntParam(cs, Props.SplatBitsValid, m_GpuEditSelected != null && m_GpuEditDeleted != null ? 1 : 0);
                uint format = (uint)m_Asset.posFormat | ((uint)m_Asset.scaleFormat << 8) | ((uint)m_Asset.shFormat << 16);
                cmb.SetComputeIntParam(cs, Props.SplatFormat, (int)format);
                cmb.SetComputeIntParam(cs, Props.SplatCount, m_SplatCount);
                cmb.SetComputeIntParam(cs, Props.SplatChunkCount, m_GpuChunksValid ? m_GpuChunks.count : 0);

                UpdateCutoutsBuffer();
                cmb.SetComputeIntParam(cs, Props.SplatCutoutsCount, m_Cutouts?.Length ?? 0);
                cmb.SetComputeBufferParam(cs, kernelIndex, Props.SplatCutouts, m_GpuEditCutouts);
            }

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixVP, matProj * matView);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixP, matProj);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeIntParams(m_CSSplatUtilities, Props.VecTileParams, new int[2] { tileCountW, tileCountH });
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatScale, m_SplatScale);
            cmb.SetComputeFloatParam(m_CSSplatUtilities, Props.SplatOpacityScale, m_OpacityScale);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOrder, m_SHOrder);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SHOnly, m_SHOnly ? 1 : 0);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcTileViewData, Props.TileSplatSortDistances, m_GpuTileSortTileDist);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcTileViewData, Props.TileSplatSortKeys, m_GpuTileSortKeys);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcTileViewData, Props.TileSplatCount, m_GpuTileSplatCount);

            var zero = new uint[1];
            /* m_GpuTileSplatCount.GetData(zero);
            Debug.Log(zero[0]);
            zero[0] = 0; */
            cmb.SetBufferData(m_GpuTileSplatCount, zero);

            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.CalcTileViewData, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcTileViewData, (m_GpuView.count + (int)gsX - 1) / (int)gsX, 1, 1);

            // Generate Indirect Buffer for the Count           
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SplatCount, m_SplatCount);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitTileSplatIndirect, Props.TileSplatCount, m_GpuTileSplatCount);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InitTileSplatIndirect, Props.TileSplatIndirect, m_GpuTileSplatIndirect);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.InitTileSplatIndirect, 1, 1, 1);
        }

        internal void SortTilePoints(CommandBuffer cmd, Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return;
            
            // sort the splats
            cmd.BeginSample(s_ProfSort);
            m_Sorter.BeforeDispatchIndirect(cmd, m_TileSorterArgs);
            // sort against distance
            m_Sorter.DispatchIndirect(cmd, m_TileSorterArgs, 24, copyIfFlip:false);
            // reorder tile ID
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTileID, Props.TileSplatSortTilesRO, m_TileSorterArgs.resources.tempKeyBuffer);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTileID, Props.TileSplatSortTiles, m_GpuTileSortTileDist);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTileID, Props.TileSplatSortKeysRO, m_TileSorterArgs.resources.tempPayloadBuffer);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTileID, Props.TileSplatSortKeys, m_GpuTileSortKeys);
            // cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ReorderTileID, Props.SplatTileViewDataRO, m_GpuView);
            cmd.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.ReorderTileID, m_GpuTileSplatIndirect, 0);
            // sort against tile ID
            Debug.Assert(kMaxTilesOnScreen < 65536);
            m_Sorter.DispatchIndirect(cmd, m_TileSorterArgs, 16);
            cmd.EndSample(s_ProfSort);
        }

        internal void CalcTileRanges(CommandBuffer cmd, Camera cam)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int tileCountW = (screenW + kSplatTileSize - 1) / kSplatTileSize;
            int tileCountH = (screenH + kSplatTileSize - 1) / kSplatTileSize;
            int tileCount = tileCountW * tileCountH;

            cmd.SetBufferData(m_GpuTileRanges, m_GpuTileRangesClearData, 0, 0, tileCount);

            cmd.SetComputeConstantBufferParam(m_CSSplatUtilities, Props.CBTileSplatCount, m_GpuTileSplatCount, 0, sizeof(uint));
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcTileRanges, Props.TileRanges, m_GpuTileRanges);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CalcTileRanges, Props.TileSplatSortTilesRO, m_GpuTileSortTileDist);
            cmd.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.CalcTileRanges, m_GpuTileSplatIndirect, 0);
        }

        internal void RenderTiles(CommandBuffer cmd, Camera cam, RTHandle target)
        {
            if (cam.cameraType == CameraType.Preview)
                return;

            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            int tileCountW = (screenW + kSplatTileSize - 1) / kSplatTileSize;
            int tileCountH = (screenH + kSplatTileSize - 1) / kSplatTileSize;

            cmd.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, new Vector4(screenW, screenH, 0, 0));
            // cmd.SetComputeIntParams(m_CSSplatUtilities, Props.VecTileParams, new int[2] { tileCountW, tileCountH });
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderTile, Props.TileRangesRO, m_GpuTileRanges);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderTile, Props.TileSplatSortKeysRO, m_GpuTileSortKeys);
            cmd.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RenderTile, Props.SplatTileViewDataRO, m_GpuView);
            cmd.SetComputeTextureParam(m_CSSplatUtilities, (int)KernelIndices.RenderTile, Props.RenderTarget, target);
            cmd.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.RenderTile, tileCountW, tileCountH, 1);
        }

        public void Update()
        {
            var curHash = m_Asset ? m_Asset.dataHash : new Hash128();
            if (m_PrevAsset != m_Asset || m_PrevHash != curHash)
            {
                m_PrevAsset = m_Asset;
                m_PrevHash = curHash;
                DisposeResourcesForAsset();
                CreateResourcesForAsset();
            }
        }

        public void ActivateCamera(int index)
        {
            Camera mainCam = Camera.main;
            if (!mainCam)
                return;
            if (!m_Asset || m_Asset.cameras == null)
                return;

            var selfTr = transform;
            var camTr = mainCam.transform;
            var prevParent = camTr.parent;
            var cam = m_Asset.cameras[index];
            camTr.parent = selfTr;
            camTr.localPosition = cam.pos;
            camTr.localRotation = Quaternion.LookRotation(cam.axisZ, cam.axisY);
            camTr.parent = prevParent;
            camTr.localScale = Vector3.one;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(camTr);
#endif
        }

        void ClearGraphicsBuffer(GraphicsBuffer buf)
        {
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.ClearBuffer, Props.DstBuffer, buf);
            m_CSSplatUtilities.SetInt(Props.BufferSize, buf.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.ClearBuffer, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.ClearBuffer, (int)((buf.count + gsX - 1) / gsX), 1, 1);
        }

        void UnionGraphicsBuffers(GraphicsBuffer dst, GraphicsBuffer src)
        {
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.OrBuffers, Props.SrcBuffer, src);
            m_CSSplatUtilities.SetBuffer((int)KernelIndices.OrBuffers, Props.DstBuffer, dst);
            m_CSSplatUtilities.SetInt(Props.BufferSize, dst.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.OrBuffers, out uint gsX, out _, out _);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.OrBuffers, (int)((dst.count + gsX - 1) / gsX), 1, 1);
        }

        static float SortableUintToFloat(uint v)
        {
            uint mask = ((v >> 31) - 1) | 0x80000000u;
            return math.asfloat(v ^ mask);
        }

        public void UpdateEditCountsAndBounds()
        {
            if (m_GpuEditSelected == null)
            {
                editSelectedSplats = 0;
                editDeletedSplats = 0;
                editCutSplats = 0;
                editModified = false;
                editSelectedBounds = default;
                return;
            }

            m_CSSplatUtilities.SetBuffer((int)KernelIndices.InitEditData, Props.DstBuffer, m_GpuEditCountsBounds);
            m_CSSplatUtilities.Dispatch((int)KernelIndices.InitEditData, 1, 1, 1);

            using CommandBuffer cmb = new CommandBuffer();
            SetAssetDataOnCS(cmb, KernelIndices.UpdateEditData);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.UpdateEditData, Props.DstBuffer, m_GpuEditCountsBounds);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)KernelIndices.UpdateEditData, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)KernelIndices.UpdateEditData, (int)((m_GpuEditSelected.count + gsX - 1) / gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);

            uint[] res = new uint[m_GpuEditCountsBounds.count];
            m_GpuEditCountsBounds.GetData(res);
            editSelectedSplats = res[0];
            editDeletedSplats = res[1];
            editCutSplats = res[2];
            Vector3 min = new Vector3(SortableUintToFloat(res[3]), SortableUintToFloat(res[4]), SortableUintToFloat(res[5]));
            Vector3 max = new Vector3(SortableUintToFloat(res[6]), SortableUintToFloat(res[7]), SortableUintToFloat(res[8]));
            Bounds bounds = default;
            bounds.SetMinMax(min, max);
            if (bounds.extents.sqrMagnitude < 0.01)
                bounds.extents = new Vector3(0.1f, 0.1f, 0.1f);
            editSelectedBounds = bounds;
        }

        void UpdateCutoutsBuffer()
        {
            int bufferSize = m_Cutouts?.Length ?? 0;
            if (bufferSize == 0)
                bufferSize = 1;
            if (m_GpuEditCutouts == null || m_GpuEditCutouts.count != bufferSize)
            {
                m_GpuEditCutouts?.Dispose();
                m_GpuEditCutouts = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, UnsafeUtility.SizeOf<GaussianCutout.ShaderData>()) { name = "GaussianCutouts" };
            }

            NativeArray<GaussianCutout.ShaderData> data = new(bufferSize, Allocator.Temp);
            if (m_Cutouts != null)
            {
                var matrix = transform.localToWorldMatrix;
                for (var i = 0; i < m_Cutouts.Length; ++i)
                {
                    data[i] = GaussianCutout.GetShaderData(m_Cutouts[i], matrix);
                }
            }

            m_GpuEditCutouts.SetData(data);
            data.Dispose();
        }

        bool EnsureEditingBuffers()
        {
            if (!HasValidAsset || !HasValidRenderSetup)
                return false;

            if (m_GpuEditSelected == null)
            {
                var target = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource |
                             GraphicsBuffer.Target.CopyDestination;
                var size = (m_SplatCount + 31) / 32;
                m_GpuEditSelected = new GraphicsBuffer(target, size, 4) { name = "GaussianSplatSelected" };
                m_GpuEditSelectedMouseDown = new GraphicsBuffer(target, size, 4) { name = "GaussianSplatSelectedInit" };
                m_GpuEditDeleted = new GraphicsBuffer(target, size, 4) { name = "GaussianSplatDeleted" };
                m_GpuEditCountsBounds = new GraphicsBuffer(target, 3 + 6, 4) { name = "GaussianSplatEditData" }; // selected count, deleted bound, cut count, float3 min, float3 max
                ClearGraphicsBuffer(m_GpuEditSelected);
                ClearGraphicsBuffer(m_GpuEditSelectedMouseDown);
                ClearGraphicsBuffer(m_GpuEditDeleted);
            }
            return m_GpuEditSelected != null;
        }

        public void EditStoreSelectionMouseDown()
        {
            if (!EnsureEditingBuffers()) return;
            Graphics.CopyBuffer(m_GpuEditSelected, m_GpuEditSelectedMouseDown);
        }

        public void EditStorePosMouseDown()
        {
            if (m_GpuEditPosMouseDown == null)
            {
                m_GpuEditPosMouseDown = new GraphicsBuffer(m_GpuPosData.target | GraphicsBuffer.Target.CopyDestination, m_GpuPosData.count, m_GpuPosData.stride) { name = "GaussianSplatEditPosMouseDown" };
            }
            Graphics.CopyBuffer(m_GpuPosData, m_GpuEditPosMouseDown);
        }
        public void EditStoreOtherMouseDown()
        {
            if (m_GpuEditOtherMouseDown == null)
            {
                m_GpuEditOtherMouseDown = new GraphicsBuffer(m_GpuOtherData.target | GraphicsBuffer.Target.CopyDestination, m_GpuOtherData.count, m_GpuOtherData.stride) { name = "GaussianSplatEditOtherMouseDown" };
            }
            Graphics.CopyBuffer(m_GpuOtherData, m_GpuEditOtherMouseDown);
        }

        public void EditUpdateSelection(Vector2 rectMin, Vector2 rectMax, Camera cam, bool subtract)
        {
            if (!EnsureEditingBuffers()) return;

            Graphics.CopyBuffer(m_GpuEditSelectedMouseDown, m_GpuEditSelected);

            var tr = transform;
            Matrix4x4 matView = cam.worldToCameraMatrix;
            Matrix4x4 matProj = GL.GetGPUProjectionMatrix(cam.projectionMatrix, true);
            Matrix4x4 matO2W = tr.localToWorldMatrix;
            Matrix4x4 matW2O = tr.worldToLocalMatrix;
            int screenW = cam.pixelWidth, screenH = cam.pixelHeight;
            Vector4 screenPar = new Vector4(screenW, screenH, 0, 0);
            Vector4 camPos = cam.transform.position;

            using var cmb = new CommandBuffer { name = "SplatSelectionUpdate" };
            SetAssetDataOnCS(cmb, KernelIndices.SelectionUpdate);

            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixVP, matProj * matView);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixMV, matView * matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixP, matProj);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, matO2W);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, matW2O);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecScreenParams, screenPar);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.VecWorldSpaceCameraPos, camPos);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_SelectionRect", new Vector4(rectMin.x, rectMax.y, rectMax.x, rectMin.y));
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.SelectionMode, subtract ? 0 : 1);

            DispatchUtilsAndExecute(cmb, KernelIndices.SelectionUpdate, m_SplatCount);
            UpdateEditCountsAndBounds();
        }

        public void EditTranslateSelection(Vector3 localSpacePosDelta)
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer { name = "SplatTranslateSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.TranslateSelection);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDelta, localSpacePosDelta);

            DispatchUtilsAndExecute(cmb, KernelIndices.TranslateSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }

        public void EditRotateSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Quaternion rotation)
        {
            if (!EnsureEditingBuffers()) return;
            if (m_GpuEditPosMouseDown == null || m_GpuEditOtherMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer { name = "SplatRotateSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.RotateSelection);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RotateSelection, Props.SplatPosMouseDown, m_GpuEditPosMouseDown);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.RotateSelection, Props.SplatOtherMouseDown, m_GpuEditOtherMouseDown);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDeltaRot, new Vector4(rotation.x, rotation.y, rotation.z, rotation.w));

            DispatchUtilsAndExecute(cmb, KernelIndices.RotateSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }


        public void EditScaleSelection(Vector3 localSpaceCenter, Matrix4x4 localToWorld, Matrix4x4 worldToLocal, Vector3 scale)
        {
            if (!EnsureEditingBuffers()) return;
            if (m_GpuEditPosMouseDown == null) return; // should have captured initial state

            using var cmb = new CommandBuffer { name = "SplatScaleSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.ScaleSelection);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ScaleSelection, Props.SplatPosMouseDown, m_GpuEditPosMouseDown);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionCenter, localSpaceCenter);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, localToWorld);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixWorldToObject, worldToLocal);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, Props.SelectionDelta, scale);

            DispatchUtilsAndExecute(cmb, KernelIndices.ScaleSelection, m_SplatCount);
            UpdateEditCountsAndBounds();
            editModified = true;
        }

        public void EditDeleteSelected()
        {
            if (!EnsureEditingBuffers()) return;
            UnionGraphicsBuffers(m_GpuEditDeleted, m_GpuEditSelected);
            EditDeselectAll();
            UpdateEditCountsAndBounds();
            if (editDeletedSplats != 0)
                editModified = true;
        }

        public void EditSelectAll()
        {
            if (!EnsureEditingBuffers()) return;
            using var cmb = new CommandBuffer { name = "SplatSelectAll" };
            SetAssetDataOnCS(cmb, KernelIndices.SelectAll);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.SelectAll, Props.DstBuffer, m_GpuEditSelected);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);
            DispatchUtilsAndExecute(cmb, KernelIndices.SelectAll, m_GpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public void EditDeselectAll()
        {
            if (!EnsureEditingBuffers()) return;
            ClearGraphicsBuffer(m_GpuEditSelected);
            UpdateEditCountsAndBounds();
        }

        public void EditInvertSelection()
        {
            if (!EnsureEditingBuffers()) return;

            using var cmb = new CommandBuffer { name = "SplatInvertSelection" };
            SetAssetDataOnCS(cmb, KernelIndices.InvertSelection);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.InvertSelection, Props.DstBuffer, m_GpuEditSelected);
            cmb.SetComputeIntParam(m_CSSplatUtilities, Props.BufferSize, m_GpuEditSelected.count);
            DispatchUtilsAndExecute(cmb, KernelIndices.InvertSelection, m_GpuEditSelected.count);
            UpdateEditCountsAndBounds();
        }

        public bool EditExportData(GraphicsBuffer dstData, bool bakeTransform)
        {
            if (!EnsureEditingBuffers()) return false;

            int flags = 0;
            var tr = transform;
            Quaternion bakeRot = tr.localRotation;
            Vector3 bakeScale = tr.localScale;

            if (bakeTransform)
                flags = 1;

            using var cmb = new CommandBuffer { name = "SplatExportData" };
            SetAssetDataOnCS(cmb, KernelIndices.ExportData);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_ExportTransformFlags", flags);
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_ExportTransformRotation", new Vector4(bakeRot.x, bakeRot.y, bakeRot.z, bakeRot.w));
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_ExportTransformScale", bakeScale);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, Props.MatrixObjectToWorld, tr.localToWorldMatrix);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.ExportData, "_ExportBuffer", dstData);

            DispatchUtilsAndExecute(cmb, KernelIndices.ExportData, m_SplatCount);
            return true;
        }

        public void EditSetSplatCount(int newSplatCount)
        {
            if (newSplatCount <= 0 || newSplatCount > GaussianSplatAsset.kMaxSplats)
            {
                Debug.LogError($"Invalid new splat count: {newSplatCount}");
                return;
            }
            if (asset.chunkData != null)
            {
                Debug.LogError("Only splats with VeryHigh quality can be resized");
                return;
            }
            if (newSplatCount == splatCount)
                return;

            int posStride = (int)(asset.posData.dataSize / asset.splatCount);
            int otherStride = (int)(asset.otherData.dataSize / asset.splatCount);
            int shStride = (int)(asset.shData.dataSize / asset.splatCount);

            // create new GPU buffers
            var newPosData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * posStride / 4, 4) { name = "GaussianPosData" };
            var newOtherData = new GraphicsBuffer(GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource, newSplatCount * otherStride / 4, 4) { name = "GaussianOtherData" };
            var newSHData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, newSplatCount * shStride / 4, 4) { name = "GaussianSHData" };

            // new texture is a RenderTexture so we can write to it from a compute shader
            var (texWidth, texHeight) = GaussianSplatAsset.CalcTextureSize(newSplatCount);
            var texFormat = GaussianSplatAsset.ColorFormatToGraphics(asset.colorFormat);
            var newColorData = new RenderTexture(texWidth, texHeight, texFormat, GraphicsFormat.None) { name = "GaussianColorData", enableRandomWrite = true };
            newColorData.Create();

            // selected/deleted buffers
            var selTarget = GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.CopySource | GraphicsBuffer.Target.CopyDestination;
            var selSize = (newSplatCount + 31) / 32;
            var newEditSelected = new GraphicsBuffer(selTarget, selSize, 4) { name = "GaussianSplatSelected" };
            var newEditSelectedMouseDown = new GraphicsBuffer(selTarget, selSize, 4) { name = "GaussianSplatSelectedInit" };
            var newEditDeleted = new GraphicsBuffer(selTarget, selSize, 4) { name = "GaussianSplatDeleted" };
            ClearGraphicsBuffer(newEditSelected);
            ClearGraphicsBuffer(newEditSelectedMouseDown);
            ClearGraphicsBuffer(newEditDeleted);

            var newGpuView = new GraphicsBuffer(GraphicsBuffer.Target.Structured, newSplatCount, kGpuViewDataSize);
            InitSortBuffers(newSplatCount);

            // copy existing data over into new buffers
            EditCopySplats(transform, newPosData, newOtherData, newSHData, newColorData, newEditDeleted, newSplatCount, 0, 0, m_SplatCount);

            // use the new buffers and the new splat count
            m_GpuPosData.Dispose();
            m_GpuOtherData.Dispose();
            m_GpuSHData.Dispose();
            DestroyImmediate(m_GpuColorData);
            m_GpuView.Dispose();

            m_GpuEditSelected?.Dispose();
            m_GpuEditSelectedMouseDown?.Dispose();
            m_GpuEditDeleted?.Dispose();

            m_GpuPosData = newPosData;
            m_GpuOtherData = newOtherData;
            m_GpuSHData = newSHData;
            m_GpuColorData = newColorData;
            m_GpuView = newGpuView;
            m_GpuEditSelected = newEditSelected;
            m_GpuEditSelectedMouseDown = newEditSelectedMouseDown;
            m_GpuEditDeleted = newEditDeleted;

            DisposeBuffer(ref m_GpuEditPosMouseDown);
            DisposeBuffer(ref m_GpuEditOtherMouseDown);

            m_SplatCount = newSplatCount;
            editModified = true;
        }

        public void EditCopySplatsInto(GaussianSplatRenderer dst, int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            EditCopySplats(
                dst.transform,
                dst.m_GpuPosData, dst.m_GpuOtherData, dst.m_GpuSHData, dst.m_GpuColorData, dst.m_GpuEditDeleted,
                dst.splatCount,
                copySrcStartIndex, copyDstStartIndex, copyCount);
            dst.editModified = true;
        }

        public void EditCopySplats(
            Transform dstTransform,
            GraphicsBuffer dstPos, GraphicsBuffer dstOther, GraphicsBuffer dstSH, Texture dstColor,
            GraphicsBuffer dstEditDeleted,
            int dstSize,
            int copySrcStartIndex, int copyDstStartIndex, int copyCount)
        {
            if (!EnsureEditingBuffers()) return;

            Matrix4x4 copyMatrix = dstTransform.worldToLocalMatrix * transform.localToWorldMatrix;
            Quaternion copyRot = copyMatrix.rotation;
            Vector3 copyScale = copyMatrix.lossyScale;

            using var cmb = new CommandBuffer { name = "SplatCopy" };
            SetAssetDataOnCS(cmb, KernelIndices.CopySplats);

            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstPos", dstPos);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstOther", dstOther);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstSH", dstSH);
            cmb.SetComputeTextureParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstColor", dstColor);
            cmb.SetComputeBufferParam(m_CSSplatUtilities, (int)KernelIndices.CopySplats, "_CopyDstEditDeleted", dstEditDeleted);

            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyDstSize", dstSize);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopySrcStartIndex", copySrcStartIndex);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyDstStartIndex", copyDstStartIndex);
            cmb.SetComputeIntParam(m_CSSplatUtilities, "_CopyCount", copyCount);

            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_CopyTransformRotation", new Vector4(copyRot.x, copyRot.y, copyRot.z, copyRot.w));
            cmb.SetComputeVectorParam(m_CSSplatUtilities, "_CopyTransformScale", copyScale);
            cmb.SetComputeMatrixParam(m_CSSplatUtilities, "_CopyTransformMatrix", copyMatrix);

            DispatchUtilsAndExecute(cmb, KernelIndices.CopySplats, copyCount);
        }

        void DispatchUtilsAndExecute(CommandBuffer cmb, KernelIndices kernel, int count)
        {
            m_CSSplatUtilities.GetKernelThreadGroupSizes((int)kernel, out uint gsX, out _, out _);
            cmb.DispatchCompute(m_CSSplatUtilities, (int)kernel, (int)((count + gsX - 1) / gsX), 1, 1);
            Graphics.ExecuteCommandBuffer(cmb);
        }

        public GraphicsBuffer GpuEditDeleted => m_GpuEditDeleted;
    }
}