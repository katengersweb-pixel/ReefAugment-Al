import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  Image as ImageIcon,
  Wand2,
  UploadCloud,
  Download,
  Trash2,
  AlertTriangle,
  CheckCircle2,
  Info,
  Loader2,
  RefreshCw,
  Sliders,
  Eye,
  X,
} from 'lucide-react';
import {
  deleteGeneratedImage,
  downloadFileFromEndpoint,
  extractErrorMessage,
  generateImageToImage,
  generateTextToImage,
  getGenerationHistory,
  getGenerationStatus,
  getImages,
  resolveMediaUrl,
  sendGeneratedToAugment,
} from '../services/api';
import { useToast } from '../components/ToastContext';
import ConfirmDialog from '../components/ConfirmDialog';
import {
  EmptyState,
  ErrorBanner,
  LoadingSpinner,
  StatusBadge,
  formatBytes,
  formatDateTime,
} from '../components/UiStates';

const SPECIES_OPTIONS = ['Fish', 'Shark', 'Turtle', 'Jellyfish', 'Dolphin', 'Coral', 'Other'];
const STYLE_OPTIONS = ['Photorealistic', 'Scientific Photography', 'Natural Underwater'];
const DIMENSION_OPTIONS = ['1024x1024', '1152x896', '896x1152', '1216x832', '832x1216', '768x768'];

export default function AIGenerationStudio() {
  const [activeTab, setActiveTab] = useState('text_to_image'); // 'text_to_image' | 'image_to_image'
  const [statusInfo, setStatusInfo] = useState(null);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [historyItems, setHistoryItems] = useState([]);
  const [loadingInit, setLoadingInit] = useState(true);
  const [initError, setInitError] = useState('');

  // Text-to-Image State
  const [prompt, setPrompt] = useState('');
  const [species, setSpecies] = useState('Fish');
  const [style, setStyle] = useState('Photorealistic');
  const [dimensions, setDimensions] = useState('1024x1024');
  const [numImages, setNumImages] = useState(1);
  const [negativePrompt, setNegativePrompt] = useState('');
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState('');

  // Image-to-Image State
  const [sourceMode, setSourceMode] = useState('upload'); // 'upload' | 'existing'
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState('');
  const [selectedExistingId, setSelectedExistingId] = useState('');
  const [i2iPrompt, setI2iPrompt] = useState('');
  const [i2iSpecies, setI2iSpecies] = useState('Fish');
  const [i2iStyle, setI2iStyle] = useState('Photorealistic');
  const [strength, setStrength] = useState(0.55);
  const [i2iNumImages, setI2iNumImages] = useState(1);
  const [i2iSteps, setI2iSteps] = useState(30);
  const [i2iGuidance, setI2iGuidance] = useState(7.5);
  const [i2iSeed, setI2iSeed] = useState('');

  // Execution & Results State
  const [generating, setGenerating] = useState(false);
  const [latestResults, setLatestResults] = useState([]);
  const [latestSourcePreview, setLatestSourcePreview] = useState(null);
  const [historyFilterMode, setHistoryFilterMode] = useState('all');
  const [previewModal, setPreviewModal] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [sendingId, setSendingId] = useState(null);

  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const toast = useToast();

  const loadStudio = async (filterMode = historyFilterMode) => {
    setLoadingInit(true);
    setInitError('');
    try {
      const [statusData, imgs, hist] = await Promise.all([
        getGenerationStatus(),
        getImages(),
        getGenerationHistory({ mode: filterMode, limit: 30 }),
      ]);
      setStatusInfo(statusData);
      setUploadedImages(imgs);
      setHistoryItems(hist.items || []);
      if (imgs.length > 0 && !selectedExistingId) {
        setSelectedExistingId(String(imgs[0].id));
      }
    } catch (err) {
      setInitError(extractErrorMessage(err, 'Failed to initialize AI Generation Studio.'));
    } finally {
      setLoadingInit(false);
    }
  };

  useEffect(() => {
    loadStudio('all');
  }, []);

  useEffect(() => {
    return () => {
      if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    };
  }, [uploadPreviewUrl]);

  const refreshHistory = async (modeOverride = historyFilterMode) => {
    try {
      const hist = await getGenerationHistory({ mode: modeOverride, limit: 30 });
      setHistoryItems(hist.items || []);
    } catch {
      // Ignore transient refresh failure
    }
  };

  const selectedExistingImage = useMemo(
    () => uploadedImages.find((img) => String(img.id) === String(selectedExistingId)) || null,
    [uploadedImages, selectedExistingId]
  );

  const handleFileChange = (file) => {
    if (!file) return;
    const valid = ['.jpg', '.jpeg', '.png', '.webp'].some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );
    if (!valid) {
      toast.error('Unsupported format. Please upload JPG, JPEG, PNG, or WEBP.', 'Invalid Image');
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      toast.error('Image file exceeds the 25 MB limit.', 'File Too Large');
      return;
    }
    if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    setUploadFile(file);
    setUploadPreviewUrl(URL.createObjectURL(file));
  };

  const handleTextToImage = async (e) => {
    e.preventDefault();
    const cleaned = prompt.trim();
    if (cleaned.length < 3) {
      toast.error('Please enter a descriptive underwater prompt (at least 3 characters).');
      return;
    }
    const [wStr, hStr] = dimensions.split('x');
    setGenerating(true);
    try {
      const res = await generateTextToImage({
        prompt: cleaned,
        species,
        style,
        width: parseInt(wStr, 10),
        height: parseInt(hStr, 10),
        num_images: numImages,
        negative_prompt: negativePrompt.trim() || undefined,
        steps,
        guidance_scale: guidanceScale,
        seed: seed.trim() !== '' ? parseInt(seed, 10) : undefined,
      });
      setLatestResults(res.images || []);
      setLatestSourcePreview(null);
      toast.success(`Generated ${res.images?.length || 0} synthetic underwater image(s) via ${res.provider}.`);
      await refreshHistory();
    } catch (err) {
      toast.error(extractErrorMessage(err, 'SDXL text-to-image generation failed.'), 'Generation Error');
      await refreshHistory();
    } finally {
      setGenerating(false);
    }
  };

  const handleImageToImage = async (e) => {
    e.preventDefault();
    if (sourceMode === 'upload' && !uploadFile) {
      toast.error('Please upload an underwater species image first.');
      return;
    }
    if (sourceMode === 'existing' && !selectedExistingId) {
      toast.error('Please select an existing uploaded image.');
      return;
    }

    const formData = new FormData();
    if (sourceMode === 'upload') {
      formData.append('file', uploadFile);
    } else {
      formData.append('image_id', selectedExistingId);
    }
    formData.append('prompt', i2iPrompt.trim());
    formData.append('species', i2iSpecies);
    formData.append('style', i2iStyle);
    formData.append('strength', String(strength));
    formData.append('num_images', String(i2iNumImages));
    formData.append('steps', String(i2iSteps));
    formData.append('guidance_scale', String(i2iGuidance));
    if (i2iSeed.trim() !== '') {
      formData.append('seed', i2iSeed.trim());
    }

    setGenerating(true);
    try {
      const res = await generateImageToImage(formData);
      setLatestResults(res.images || []);
      setLatestSourcePreview(res.source_image || null);
      toast.success(`Generated ${res.images?.length || 0} synthetic variation(s) from source image.`);
      const imgs = await getImages();
      setUploadedImages(imgs);
      await refreshHistory();
    } catch (err) {
      toast.error(extractErrorMessage(err, 'SDXL image-to-image generation failed.'), 'Generation Error');
      await refreshHistory();
    } finally {
      setGenerating(false);
    }
  };

  const handleSendToAugment = async (genId) => {
    setSendingId(genId);
    try {
      const uploaded = await sendGeneratedToAugment(genId);
      toast.success('Synthetic image added to dataset. Opening Augmentation Studio...');
      navigate(`/studio?imageId=${uploaded.id}`);
    } catch (err) {
      toast.error(extractErrorMessage(err, 'Unable to pass image to Augmentation Studio.'));
    } finally {
      setSendingId(null);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await deleteGeneratedImage(deleteTarget.id);
      toast.success('Deleted generated image record.');
      setLatestResults((prev) => prev.filter((item) => item.id !== deleteTarget.id));
      setDeleteTarget(null);
      await refreshHistory();
    } catch (err) {
      toast.error(extractErrorMessage(err, 'Failed to delete generated image.'));
    } finally {
      setDeleting(false);
    }
  };

  if (loadingInit) return <LoadingSpinner label="Loading AI Image Generation Studio..." />;
  if (initError) return <ErrorBanner message={initError} onRetry={() => loadStudio(historyFilterMode)} />;

  const activeSourceUrl =
    sourceMode === 'upload'
      ? uploadPreviewUrl
      : selectedExistingImage
      ? resolveMediaUrl(selectedExistingImage.image_url)
      : '';

  return (
    <div className="space-y-6">
      {/* Synthetic AI Safety Notice */}
      <div className="rounded-2xl border border-cyan-200 bg-gradient-to-r from-[#041428] via-[#092b4c] to-[#0c4a6e] p-5 text-white shadow-md">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-400/20 px-3 py-0.5 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-300/30">
                <Sparkles className="w-3.5 h-3.5" />
                Stable Diffusion XL (SDXL)
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-amber-400/20 px-2.5 py-0.5 text-[11px] font-semibold text-amber-200 ring-1 ring-amber-300/30">
                Synthetic Data Only
              </span>
            </div>
            <h2 className="text-lg font-bold tracking-tight">AI Underwater Image Generation Studio</h2>
            <p className="text-xs text-cyan-100/85 max-w-3xl leading-relaxed">
              Generate synthetic underwater marine images via hosted GPU SDXL inference for dataset diversity. Outputs are labeled as synthetic and are not scientifically verified species photographs.
            </p>
          </div>

          {statusInfo && (
            <div className="rounded-xl bg-white/10 border border-white/15 px-4 py-3 text-xs space-y-1 shrink-0">
              <div className="flex items-center gap-2">
                {statusInfo.configured ? (
                  <CheckCircle2 className="w-4 h-4 text-teal-300 shrink-0" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-amber-300 shrink-0" />
                )}
                <span className="font-semibold">
                  {statusInfo.configured ? 'Inference Provider Configured' : 'API Credentials Required'}
                </span>
              </div>
              <p className="text-cyan-100/80 font-mono text-[11px]">
                {statusInfo.provider} • {statusInfo.model}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Credential / Capability Notice if Unconfigured */}
      {statusInfo && !statusInfo.configured && (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-amber-950 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1.5">
            <p className="font-bold">Hosted SDXL Inference Credentials Not Configured</p>
            <p className="text-amber-900 leading-relaxed">{statusInfo.message}</p>
            <p className="font-mono text-[11px] bg-white/80 border border-amber-200 rounded-lg px-3 py-2 text-slate-800">
              Required backend env vars: {statusInfo.required_env_vars.join(', ')}
            </p>
          </div>
        </div>
      )}

      {/* Mode Tabs */}
      <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
          <div className="inline-flex rounded-xl bg-slate-100 p-1">
            <button
              type="button"
              onClick={() => setActiveTab('text_to_image')}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition cursor-pointer ${
                activeTab === 'text_to_image'
                  ? 'bg-white text-cyan-800 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Sparkles className="w-4 h-4 text-cyan-600" />
              Text-to-Image (SDXL)
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('image_to_image')}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition cursor-pointer ${
                activeTab === 'image_to_image'
                  ? 'bg-white text-cyan-800 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <ImageIcon className="w-4 h-4 text-teal-600" />
              Image-to-Image (SDXL)
            </button>
          </div>

          <button
            type="button"
            onClick={() => loadStudio(historyFilterMode)}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh Status
          </button>
        </div>

        {/* TAB 1: TEXT-TO-IMAGE */}
        {activeTab === 'text_to_image' ? (
          <form onSubmit={handleTextToImage} className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7 space-y-4">
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">
                  Underwater Scene & Species Prompt
                </label>
                <textarea
                  rows={4}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the underwater species and environment (e.g., vibrant clownfish swimming near sea anemone with sunlight rays filtering through clear reef water)..."
                  className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-2.5 text-sm text-slate-900 focus:bg-white focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="block text-xs font-bold text-slate-700">Species Category</label>
                  <select
                    value={species}
                    onChange={(e) => setSpecies(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-medium text-slate-800 focus:bg-white focus:border-cyan-500 focus:outline-none"
                  >
                    {SPECIES_OPTIONS.map((sp) => (
                      <option key={sp} value={sp}>
                        {sp}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-bold text-slate-700">Visual Style</label>
                  <select
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-medium text-slate-800 focus:bg-white focus:border-cyan-500 focus:outline-none"
                  >
                    {STYLE_OPTIONS.map((st) => (
                      <option key={st} value={st}>
                        {st}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">
                  Negative Prompt (Optional)
                </label>
                <input
                  type="text"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="Optional exclusions (default excludes watermarks, text, extra fins, heavy blur)"
                  className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-2 text-xs text-slate-800 focus:bg-white focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="lg:col-span-5 rounded-2xl border border-slate-200 bg-slate-50/70 p-4 space-y-4">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-800">
                <Sliders className="w-4 h-4 text-cyan-600" />
                SDXL Generation Settings
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="block text-xs font-semibold text-slate-700">Dimensions</label>
                  <select
                    value={dimensions}
                    onChange={(e) => setDimensions(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800"
                  >
                    {DIMENSION_OPTIONS.map((dim) => (
                      <option key={dim} value={dim}>
                        {dim}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="block text-xs font-semibold text-slate-700">Number of Images</label>
                  <select
                    value={numImages}
                    onChange={(e) => setNumImages(Number(e.target.value))}
                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800"
                  >
                    {[1, 2, 3, 4].map((n) => (
                      <option key={n} value={n}>
                        {n} image{n > 1 ? 's' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-slate-700">Inference Steps</span>
                  <span className="font-mono font-bold text-cyan-800">{steps}</span>
                </div>
                <input
                  type="range"
                  min={15}
                  max={50}
                  step={1}
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-slate-700">Guidance Scale (CFG)</span>
                  <span className="font-mono font-bold text-cyan-800">{guidanceScale}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={15}
                  step={0.5}
                  value={guidanceScale}
                  onChange={(e) => setGuidanceScale(Number(e.target.value))}
                  className="w-full cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-slate-700">Seed (Optional)</label>
                <input
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="Random if blank"
                  className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-800"
                />
              </div>

              <button
                type="submit"
                disabled={generating}
                className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-600 to-teal-600 px-5 py-3 text-xs font-bold text-white shadow-sm hover:from-cyan-700 hover:to-teal-700 transition cursor-pointer disabled:opacity-60"
              >
                {generating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Synthesizing with SDXL...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Generate Underwater Images
                  </>
                )}
              </button>
            </div>
          </form>
        ) : (
          /* TAB 2: IMAGE-TO-IMAGE */
          <form onSubmit={handleImageToImage} className="space-y-5">
            {statusInfo && statusInfo.configured && !statusInfo.supports_image_to_image && (
              <div className="rounded-xl border border-cyan-200 bg-cyan-50 p-3.5 text-xs text-cyan-900 flex items-start gap-2.5">
                <Info className="w-4 h-4 text-cyan-700 shrink-0 mt-0.5" />
                <span>{statusInfo.message}</span>
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Source Image Selection / Upload */}
              <div className="lg:col-span-5 space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-slate-700">1. Source Underwater Image</label>
                  <div className="inline-flex rounded-lg bg-slate-100 p-0.5 text-[11px]">
                    <button
                      type="button"
                      onClick={() => setSourceMode('upload')}
                      className={`px-2.5 py-1 rounded-md font-semibold cursor-pointer ${
                        sourceMode === 'upload' ? 'bg-white text-cyan-800 shadow-2xs' : 'text-slate-600'
                      }`}
                    >
                      Upload New
                    </button>
                    <button
                      type="button"
                      onClick={() => setSourceMode('existing')}
                      className={`px-2.5 py-1 rounded-md font-semibold cursor-pointer ${
                        sourceMode === 'existing' ? 'bg-white text-cyan-800 shadow-2xs' : 'text-slate-600'
                      }`}
                    >
                      From Dataset ({uploadedImages.length})
                    </button>
                  </div>
                </div>

                {sourceMode === 'upload' ? (
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50/70 p-5 text-center hover:border-cyan-400 cursor-pointer transition"
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".jpg,.jpeg,.png,.webp"
                      className="hidden"
                      onChange={(e) => {
                        handleFileChange(e.target.files?.[0]);
                        e.target.value = '';
                      }}
                    />
                    {uploadPreviewUrl ? (
                      <div className="space-y-2">
                        <img
                          src={uploadPreviewUrl}
                          alt="Source preview"
                          className="mx-auto max-h-44 rounded-xl object-contain"
                        />
                        <p className="text-xs font-semibold text-slate-800 truncate">{uploadFile?.name}</p>
                        <p className="text-[11px] text-cyan-700">Click to replace source image</p>
                      </div>
                    ) : (
                      <div className="py-4 space-y-2">
                        <UploadCloud className="w-8 h-8 mx-auto text-cyan-600" />
                        <p className="text-xs font-semibold text-slate-800">
                          Click to upload underwater image (JPG, PNG, WEBP)
                        </p>
                        <p className="text-[11px] text-slate-500">Original upload is always preserved</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-3">
                    <select
                      value={selectedExistingId}
                      onChange={(e) => setSelectedExistingId(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-medium text-slate-800"
                    >
                      {uploadedImages.length === 0 ? (
                        <option value="">No uploaded images in dataset</option>
                      ) : (
                        uploadedImages.map((img) => (
                          <option key={img.id} value={img.id}>
                            {img.original_filename} ({img.width}x{img.height})
                          </option>
                        ))
                      )}
                    </select>
                    {activeSourceUrl && (
                      <div className="rounded-xl border border-slate-200 bg-slate-50 p-2 flex justify-center">
                        <img
                          src={activeSourceUrl}
                          alt="Selected source"
                          className="max-h-44 rounded-lg object-contain"
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Modification Prompt & SDXL Img2Img Parameters */}
              <div className="lg:col-span-7 rounded-2xl border border-slate-200 bg-slate-50/70 p-4 space-y-4">
                <div className="space-y-1.5">
                  <label className="block text-xs font-bold text-slate-700">
                    2. Modification Prompt (Optional)
                  </label>
                  <textarea
                    rows={2}
                    value={i2iPrompt}
                    onChange={(e) => setI2iPrompt(e.target.value)}
                    placeholder="Optional guidance (e.g., deeper reef lighting, clearer water visibility, coral background variation)..."
                    className="w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-900 focus:border-cyan-500 focus:outline-none"
                  />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="space-y-1">
                    <label className="block text-xs font-semibold text-slate-700">Species Context</label>
                    <select
                      value={i2iSpecies}
                      onChange={(e) => setI2iSpecies(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800"
                    >
                      {SPECIES_OPTIONS.map((sp) => (
                        <option key={sp} value={sp}>
                          {sp}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs font-semibold text-slate-700">Visual Style</label>
                    <select
                      value={i2iStyle}
                      onChange={(e) => setI2iStyle(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800"
                    >
                      {STYLE_OPTIONS.map((st) => (
                        <option key={st} value={st}>
                          {st}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs font-semibold text-slate-700">Variations</label>
                    <select
                      value={i2iNumImages}
                      onChange={(e) => setI2iNumImages(Number(e.target.value))}
                      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800"
                    >
                      {[1, 2, 3, 4].map((n) => (
                        <option key={n} value={n}>
                          {n} variation{n > 1 ? 's' : ''}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-semibold text-slate-700">Transformation Strength</span>
                    <span className="font-mono font-bold text-cyan-800">{strength}</span>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={0.95}
                    step={0.05}
                    value={strength}
                    onChange={(e) => setStrength(Number(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                  <div className="flex justify-between text-[11px] text-slate-500">
                    <span>0.10 (Preserve original tightly)</span>
                    <span>0.95 (Strong synthetic change)</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-semibold text-slate-700">Steps</span>
                      <span className="font-mono font-bold text-cyan-800">{i2iSteps}</span>
                    </div>
                    <input
                      type="range"
                      min={15}
                      max={50}
                      value={i2iSteps}
                      onChange={(e) => setI2iSteps(Number(e.target.value))}
                      className="w-full cursor-pointer"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-semibold text-slate-700">Guidance</span>
                      <span className="font-mono font-bold text-cyan-800">{i2iGuidance}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={15}
                      step={0.5}
                      value={i2iGuidance}
                      onChange={(e) => setI2iGuidance(Number(e.target.value))}
                      className="w-full cursor-pointer"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs font-semibold text-slate-700">Seed (Optional)</label>
                    <input
                      type="number"
                      min={0}
                      value={i2iSeed}
                      onChange={(e) => setI2iSeed(e.target.value)}
                      placeholder="Random"
                      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-800"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={generating}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-600 to-teal-600 px-5 py-3 text-xs font-bold text-white shadow-sm hover:from-cyan-700 hover:to-teal-700 transition cursor-pointer disabled:opacity-60"
                >
                  {generating ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating SDXL Variations...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Generate Image-to-Image Variations
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>
        )}
      </div>

      {/* Latest Generated Preview Section (Side-by-Side for Image-to-Image + Grid) */}
      {latestResults.length > 0 && (
        <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Latest Generated Synthetic Outputs ({latestResults.length})
              </h3>
              <p className="text-xs text-slate-500">
                Saved to outputs/ and Results Gallery • Synthetic SDXL generation
              </p>
            </div>
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 border border-amber-200 px-3 py-1 text-[11px] font-semibold text-amber-800">
              Synthetic AI-Generated • Not a verified specimen photo
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {latestSourcePreview && (
              <div className="rounded-2xl border-2 border-slate-300 bg-slate-50 overflow-hidden flex flex-col">
                <div className="aspect-square bg-slate-200 relative">
                  <img
                    src={resolveMediaUrl(latestSourcePreview.image_url)}
                    alt={latestSourcePreview.original_filename}
                    className="w-full h-full object-cover"
                  />
                  <span className="absolute top-2.5 left-2.5 rounded-md bg-slate-900/80 px-2 py-0.5 text-[10px] font-bold text-white">
                    ORIGINAL SOURCE
                  </span>
                </div>
                <div className="p-3 text-xs">
                  <p className="font-bold text-slate-900 truncate">{latestSourcePreview.original_filename}</p>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    {latestSourcePreview.width}x{latestSourcePreview.height}px • Preserved upload
                  </p>
                </div>
              </div>
            )}

            {latestResults.map((item) => (
              <div
                key={item.id}
                className="rounded-2xl border border-cyan-200 bg-white overflow-hidden shadow-2xs flex flex-col"
              >
                <div className="aspect-square bg-slate-100 relative group overflow-hidden">
                  <img
                    src={resolveMediaUrl(item.image_url)}
                    alt={item.output_filename}
                    onClick={() => setPreviewModal(item)}
                    className="w-full h-full object-cover cursor-pointer group-hover:scale-105 transition duration-200"
                  />
                  <span className="absolute top-2.5 left-2.5 rounded-md bg-amber-500 px-2 py-0.5 text-[10px] font-bold text-slate-950 shadow">
                    SYNTHETIC AI
                  </span>
                </div>
                <div className="p-3 flex-1 flex flex-col justify-between gap-2.5">
                  <div>
                    <p className="text-xs font-bold text-slate-900 truncate">{item.output_filename}</p>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      {item.species} • {item.style} • {item.width}x{item.height}px
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 pt-2 border-t border-slate-100">
                    <button
                      type="button"
                      onClick={() => handleSendToAugment(item.id)}
                      disabled={sendingId === item.id}
                      className="flex-1 inline-flex items-center justify-center gap-1 rounded-xl bg-cyan-50 px-2.5 py-1.5 text-xs font-semibold text-cyan-800 hover:bg-cyan-100 cursor-pointer"
                    >
                      <Wand2 className="w-3.5 h-3.5" />
                      {sendingId === item.id ? 'Opening...' : 'Augment (15 Ops)'}
                    </button>
                    <button
                      type="button"
                      onClick={() => downloadFileFromEndpoint(item.download_url, item.output_filename)}
                      className="p-1.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-100 cursor-pointer"
                      title="Download PNG"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Generation History */}
      <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">AI Generation History ({historyItems.length})</h3>
            <p className="text-xs text-slate-500">Persistent SQLite log of SDXL generations</p>
          </div>
          <select
            value={historyFilterMode}
            onChange={(e) => {
              setHistoryFilterMode(e.target.value);
              refreshHistory(e.target.value);
            }}
            className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-800"
          >
            <option value="all">All Modes</option>
            <option value="text_to_image">Text-to-Image</option>
            <option value="image_to_image">Image-to-Image</option>
          </select>
        </div>

        {historyItems.length === 0 ? (
          <EmptyState
            icon={Sparkles}
            title="No AI generations yet"
            description="Configure your hosted SDXL provider credentials and generate underwater species images above."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="py-2.5 px-3">Preview</th>
                  <th className="py-2.5 px-3">Mode</th>
                  <th className="py-2.5 px-3">Species / Style</th>
                  <th className="py-2.5 px-3">Prompt</th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Created</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs">
                {historyItems.map((row) => (
                  <tr key={row.id} className="hover:bg-slate-50/70">
                    <td className="py-2.5 px-3">
                      {row.image_url ? (
                        <img
                          src={resolveMediaUrl(row.image_url)}
                          alt={row.output_filename}
                          onClick={() => setPreviewModal(row)}
                          className="w-12 h-12 rounded-lg object-cover bg-slate-200 cursor-pointer"
                        />
                      ) : (
                        <div className="w-12 h-12 rounded-lg bg-slate-100 flex items-center justify-center text-slate-400">
                          <AlertTriangle className="w-4 h-4 text-rose-500" />
                        </div>
                      )}
                    </td>
                    <td className="py-2.5 px-3 whitespace-nowrap">
                      <span className="rounded-md bg-cyan-50 px-2 py-0.5 text-[11px] font-semibold text-cyan-800 border border-cyan-100">
                        {row.mode === 'text_to_image' ? 'Text-to-Image' : 'Image-to-Image'}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 whitespace-nowrap">
                      <span className="font-semibold text-slate-800">{row.species}</span>
                      <span className="block text-[11px] text-slate-500">{row.style}</span>
                    </td>
                    <td className="py-2.5 px-3 max-w-xs">
                      <p className="truncate text-slate-800" title={row.prompt}>
                        {row.prompt}
                      </p>
                      {row.error_message && (
                        <p className="text-[11px] text-rose-600 truncate" title={row.error_message}>
                          {row.error_message}
                        </p>
                      )}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-[11px] text-slate-600">
                      {row.model}
                    </td>
                    <td className="py-2.5 px-3">
                      <StatusBadge status={row.status} />
                    </td>
                    <td className="py-2.5 px-3 text-slate-500 whitespace-nowrap">
                      {formatDateTime(row.created_at)}
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center justify-end gap-1.5">
                        {row.image_url && (
                          <button
                            type="button"
                            onClick={() => setPreviewModal(row)}
                            className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-100 cursor-pointer"
                            title="Preview"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {row.status === 'success' && (
                          <button
                            type="button"
                            onClick={() => handleSendToAugment(row.id)}
                            disabled={sendingId === row.id}
                            className="p-1.5 rounded-lg border border-cyan-200 text-cyan-700 hover:bg-cyan-50 cursor-pointer"
                            title="Augment with 15 techniques"
                          >
                            <Wand2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                        {row.download_url && (
                          <button
                            type="button"
                            onClick={() => downloadFileFromEndpoint(row.download_url, row.output_filename)}
                            className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-100 cursor-pointer"
                            title="Download"
                          >
                            <Download className="w-3.5 h-3.5" />
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => setDeleteTarget(row)}
                          className="p-1.5 rounded-lg border border-rose-200 text-rose-600 hover:bg-rose-50 cursor-pointer"
                          title="Delete"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Preview Lightbox Modal */}
      {previewModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 backdrop-blur-xs p-4"
          onClick={() => setPreviewModal(null)}
        >
          <div
            className="max-w-3xl w-full rounded-2xl bg-white shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-200">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold text-slate-900">{previewModal.output_filename}</h3>
                  <span className="rounded-md bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-900">
                    SYNTHETIC AI
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  {previewModal.model} • {previewModal.species} ({previewModal.style})
                </p>
              </div>
              <button
                type="button"
                onClick={() => setPreviewModal(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="bg-slate-950 p-4 flex items-center justify-center max-h-[65vh]">
              <img
                src={resolveMediaUrl(previewModal.image_url)}
                alt={previewModal.output_filename}
                className="max-h-[60vh] object-contain rounded-lg"
              />
            </div>
            <div className="p-4 bg-slate-50 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="text-slate-600">
                {previewModal.width}x{previewModal.height}px • {formatBytes(previewModal.file_size)}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => handleSendToAugment(previewModal.id)}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-50 border border-cyan-200 px-3.5 py-2 font-semibold text-cyan-800 hover:bg-cyan-100 cursor-pointer"
                >
                  <Wand2 className="w-3.5 h-3.5" />
                  Augment (15 Techniques)
                </button>
                <button
                  type="button"
                  onClick={() => downloadFileFromEndpoint(previewModal.download_url, previewModal.output_filename)}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-600 px-4 py-2 font-semibold text-white hover:bg-cyan-700 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Delete Generated Image?"
        message={`This removes '${deleteTarget?.output_filename || `Generation #${deleteTarget?.id}`}' from storage and generation history. Original uploaded images remain untouched.`}
        confirmLabel="Delete Image"
        loading={deleting}
        onConfirm={handleConfirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
