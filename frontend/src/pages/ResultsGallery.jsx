import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Search,
  Download,
  Trash2,
  Eye,
  SplitSquareHorizontal,
  Images,
  CheckSquare,
  Square,
  Archive,
  X,
  ChevronLeft,
  ChevronRight,
  Wand2,
} from 'lucide-react';
import {
  deleteGeneratedImage,
  deleteResult,
  downloadFileFromEndpoint,
  exportSelectedZip,
  extractErrorMessage,
  getResults,
  getTechniques,
  resolveMediaUrl,
  sendGeneratedToAugment,
} from '../services/api';
import { useToast } from '../components/ToastContext';
import ConfirmDialog from '../components/ConfirmDialog';
import {
  EmptyState,
  ErrorBanner,
  LoadingSpinner,
  formatBytes,
  formatDateTime,
} from '../components/UiStates';

const PAGE_SIZE = 12;

export default function ResultsGallery() {
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [techniques, setTechniques] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [search, setSearch] = useState('');
  const [techniqueFilter, setTechniqueFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState('newest');
  const [offset, setOffset] = useState(0);

  const [selectedKeys, setSelectedKeys] = useState([]);
  const [previewItem, setPreviewItem] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const navigate = useNavigate();
  const toast = useToast();

  const getItemKey = (item) => `${item.origin_type || 'augmented'}_${item.id}`;

  const fetchResults = async (opts = {}) => {
    setLoading(true);
    setError('');
    try {
      const currentOffset = opts.offset !== undefined ? opts.offset : offset;
      const data = await getResults({
        search: opts.search !== undefined ? opts.search : search,
        technique: opts.technique !== undefined ? opts.technique : techniqueFilter,
        sort: opts.sort !== undefined ? opts.sort : sortOrder,
        limit: PAGE_SIZE,
        offset: currentOffset,
      });
      setResults(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(extractErrorMessage(err, 'Failed to fetch generated images.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const init = async () => {
      try {
        const techs = await getTechniques();
        setTechniques(techs);
      } catch {
        // Ignore non-fatal filter list error
      }
      await fetchResults({ offset: 0 });
    };
    init();
  }, []);

  const handleFilterChange = (nextSearch, nextTech, nextSort) => {
    setSearch(nextSearch);
    setTechniqueFilter(nextTech);
    setSortOrder(nextSort);
    setOffset(0);
    fetchResults({ search: nextSearch, technique: nextTech, sort: nextSort, offset: 0 });
  };

  const toggleSelect = (item) => {
    const key = getItemKey(item);
    setSelectedKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const isAi = deleteTarget.origin_type === 'ai_generated';
      if (isAi) {
        await deleteGeneratedImage(deleteTarget.generation_id || deleteTarget.id);
      } else {
        await deleteResult(deleteTarget.id);
      }
      toast.success(`Deleted '${deleteTarget.output_filename}'. Original image remains preserved.`);
      const targetKey = getItemKey(deleteTarget);
      setSelectedKeys((prev) => prev.filter((k) => k !== targetKey));
      setDeleteTarget(null);
      await fetchResults();
    } catch (err) {
      toast.error(extractErrorMessage(err, 'Failed to delete result.'));
    } finally {
      setDeleting(false);
    }
  };

  const handleSendToAugment = async (item) => {
    try {
      const uploaded = await sendGeneratedToAugment(item.generation_id || item.id);
      toast.success('Synthetic image added to dataset. Opening Augmentation Studio...');
      navigate(`/studio?imageId=${uploaded.id}`);
    } catch (err) {
      toast.error(extractErrorMessage(err, 'Failed to pass image into Augmentation Studio.'));
    }
  };

  const handleExportSelected = async () => {
    if (selectedKeys.length === 0) return;
    const resultIds = [];
    const generatedIds = [];
    selectedKeys.forEach((key) => {
      if (key.startsWith('ai_generated_')) {
        generatedIds.push(Number(key.replace('ai_generated_', '')));
      } else {
        resultIds.push(Number(key.replace('augmented_', '')));
      }
    });
    try {
      await exportSelectedZip({ resultIds, generatedIds });
      toast.success(`Exported ${selectedKeys.length} selected image(s) as ZIP.`);
    } catch (err) {
      toast.error(extractErrorMessage(err, 'Failed to export selected images.'));
    }
  };

  const handleExportAll = async () => {
    try {
      await downloadFileFromEndpoint('/api/export/zip', 'reefaugment_all_results.zip');
      toast.success('Exported all generated results as ZIP with manifest.json.');
    } catch (err) {
      toast.error(extractErrorMessage(err, 'No generated images available to export.'));
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div className="space-y-6">
      {/* Filter & Export Toolbar */}
      <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3 flex-1">
            <div className="relative flex-1 min-w-[200px] max-w-xs">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={search}
                onChange={(e) => handleFilterChange(e.target.value, techniqueFilter, sortOrder)}
                placeholder="Search output or original filename..."
                className="w-full rounded-xl border border-slate-200 bg-slate-50 pl-9 pr-3 py-2 text-xs text-slate-800 focus:bg-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <select
              value={techniqueFilter}
              onChange={(e) => handleFilterChange(search, e.target.value, sortOrder)}
              className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-800 focus:border-cyan-500 focus:outline-none"
            >
              <option value="all">All Techniques & AI</option>
              <option value="ai_generated">AI Generated (All SDXL)</option>
              <option value="sdxl_text_to_image">SDXL Text-to-Image (Synthetic)</option>
              <option value="sdxl_image_to_image">SDXL Image-to-Image (Synthetic)</option>
              {techniques.map((t) => (
                <option key={t.key} value={t.key}>
                  {t.name}
                </option>
              ))}
            </select>

            <select
              value={sortOrder}
              onChange={(e) => handleFilterChange(search, techniqueFilter, e.target.value)}
              className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium text-slate-800 focus:border-cyan-500 focus:outline-none"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {selectedKeys.length > 0 && (
              <button
                type="button"
                onClick={handleExportSelected}
                className="inline-flex items-center gap-1.5 rounded-xl bg-teal-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-teal-700 cursor-pointer"
              >
                <Archive className="w-3.5 h-3.5" />
                Export Selected ({selectedKeys.length})
              </button>
            )}
            <button
              type="button"
              onClick={handleExportAll}
              disabled={total === 0}
              className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-[#07213e] to-[#0a4b6e] px-3.5 py-2 text-xs font-semibold text-white hover:from-[#0a2d52] hover:to-[#0e5f8a] cursor-pointer disabled:opacity-50"
            >
              <Download className="w-3.5 h-3.5 text-cyan-300" />
              Export All ZIP ({total})
            </button>
          </div>
        </div>
      </div>

      {/* Gallery Grid */}
      {loading ? (
        <LoadingSpinner label="Loading generated dataset..." />
      ) : error ? (
        <ErrorBanner message={error} onRetry={() => fetchResults()} />
      ) : results.length === 0 ? (
        <EmptyState
          icon={Images}
          title="No augmented images match your filters"
          description="Generate images in the Augmentation Studio or Batch Processing to populate your Results Gallery."
          action={
            <Link
              to="/studio"
              className="inline-flex items-center gap-2 rounded-xl bg-cyan-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-cyan-700"
            >
              Open Augmentation Studio
            </Link>
          }
        />
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {results.map((item) => {
              const itemKey = getItemKey(item);
              const isSelected = selectedKeys.includes(itemKey);
              const isAi = item.origin_type === 'ai_generated' || item.is_synthetic;
              return (
                <div
                  key={itemKey}
                  className={`rounded-2xl border bg-white overflow-hidden shadow-2xs hover:shadow-md transition flex flex-col ${
                    isSelected ? 'border-cyan-500 ring-2 ring-cyan-500/20' : 'border-slate-200'
                  }`}
                >
                  <div className="aspect-[4/3] bg-slate-100 relative group overflow-hidden">
                    <img
                      src={resolveMediaUrl(item.image_url)}
                      alt={item.output_filename}
                      onClick={() => setPreviewItem(item)}
                      className="w-full h-full object-cover cursor-pointer group-hover:scale-105 transition duration-200"
                    />
                    <button
                      type="button"
                      onClick={() => toggleSelect(item)}
                      className="absolute top-2.5 left-2.5 rounded-lg bg-white/90 p-1 text-cyan-700 shadow cursor-pointer"
                      title="Select for ZIP export"
                    >
                      {isSelected ? (
                        <CheckSquare className="w-4 h-4" />
                      ) : (
                        <Square className="w-4 h-4 text-slate-400" />
                      )}
                    </button>
                    {isAi && (
                      <span className="absolute top-2.5 right-2.5 rounded-md bg-amber-500 px-2 py-0.5 text-[10px] font-bold text-slate-950 shadow">
                        SYNTHETIC AI
                      </span>
                    )}
                    <span className="absolute bottom-2.5 left-2.5 rounded-md bg-slate-950/75 px-2 py-0.5 text-[10px] font-semibold text-cyan-200">
                      {item.technique_label}
                    </span>
                  </div>

                  <div className="p-3.5 flex-1 flex flex-col justify-between gap-2.5">
                    <div>
                      <p className="text-xs font-bold text-slate-900 truncate" title={item.output_filename}>
                        {item.output_filename}
                      </p>
                      <p className="text-[11px] text-slate-500 truncate mt-0.5" title={item.prompt || item.original_filename || ''}>
                        {isAi && item.prompt ? (
                          <>
                            Prompt: <span className="font-medium text-slate-700">{item.prompt}</span>
                          </>
                        ) : (
                          <>
                            Source: <span className="font-medium text-slate-700">{item.original_filename || '—'}</span>
                          </>
                        )}
                      </p>
                      <p className="text-[11px] text-slate-400 mt-1">
                        {item.width}x{item.height}px • {formatBytes(item.file_size)}
                      </p>
                      <p className="text-[11px] text-slate-400">{formatDateTime(item.created_at)}</p>
                    </div>

                    <div className="flex items-center gap-1.5 pt-2 border-t border-slate-100">
                      <button
                        type="button"
                        onClick={() => setPreviewItem(item)}
                        className="p-1.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-100 cursor-pointer"
                        title="Preview image"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                      {isAi ? (
                        <button
                          type="button"
                          onClick={() => handleSendToAugment(item)}
                          className="flex-1 inline-flex items-center justify-center gap-1 rounded-xl bg-cyan-50 px-2.5 py-1.5 text-xs font-semibold text-cyan-800 hover:bg-cyan-100 cursor-pointer"
                        >
                          <Wand2 className="w-3.5 h-3.5" />
                          Augment
                        </button>
                      ) : (
                        <Link
                          to={`/comparison?originalId=${item.original_image_id}&resultId=${item.id}`}
                          className="flex-1 inline-flex items-center justify-center gap-1 rounded-xl bg-cyan-50 px-2.5 py-1.5 text-xs font-semibold text-cyan-800 hover:bg-cyan-100"
                        >
                          <SplitSquareHorizontal className="w-3.5 h-3.5" />
                          Compare
                        </Link>
                      )}
                      <button
                        type="button"
                        onClick={() => downloadFileFromEndpoint(item.download_url, item.output_filename)}
                        className="p-1.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-100 cursor-pointer"
                        title="Download generated image"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => setDeleteTarget(item)}
                        className="p-1.5 rounded-xl border border-rose-200 text-rose-600 hover:bg-rose-50 cursor-pointer"
                        title="Delete generated image"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div className="flex items-center justify-between rounded-2xl bg-white px-5 py-3.5 shadow-xs ring-1 ring-slate-200/80 text-xs">
              <span className="text-slate-600">
                Showing <strong>{offset + 1}</strong>–<strong>{Math.min(offset + PAGE_SIZE, total)}</strong> of{' '}
                <strong>{total}</strong> generated images
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={offset === 0}
                  onClick={() => {
                    const next = Math.max(0, offset - PAGE_SIZE);
                    setOffset(next);
                    fetchResults({ offset: next });
                  }}
                  className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-1.5 font-semibold text-slate-700 hover:bg-slate-50 cursor-pointer disabled:opacity-40"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  Prev
                </button>
                <span className="font-mono text-slate-600 px-2">
                  Page {currentPage} / {totalPages}
                </span>
                <button
                  type="button"
                  disabled={offset + PAGE_SIZE >= total}
                  onClick={() => {
                    const next = offset + PAGE_SIZE;
                    setOffset(next);
                    fetchResults({ offset: next });
                  }}
                  className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-1.5 font-semibold text-slate-700 hover:bg-slate-50 cursor-pointer disabled:opacity-40"
                >
                  Next
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Lightbox Preview Modal */}
      {previewItem && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 backdrop-blur-xs p-4"
          onClick={() => setPreviewItem(null)}
        >
          <div
            className="max-w-3xl w-full rounded-2xl bg-white shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-200">
              <div>
                <h3 className="text-sm font-bold text-slate-900">{previewItem.output_filename}</h3>
                <p className="text-xs text-slate-500">
                  {previewItem.technique_label} • Original: {previewItem.original_filename}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setPreviewItem(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="bg-slate-950 p-4 flex items-center justify-center max-h-[65vh]">
              <img
                src={resolveMediaUrl(previewItem.image_url)}
                alt={previewItem.output_filename}
                className="max-h-[60vh] object-contain rounded-lg"
              />
            </div>
            <div className="p-4 bg-slate-50 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="text-slate-600">
                {previewItem.width}x{previewItem.height}px • {formatBytes(previewItem.file_size)}
              </div>
              <button
                type="button"
                onClick={() => downloadFileFromEndpoint(previewItem.download_url, previewItem.output_filename)}
                className="inline-flex items-center gap-1.5 rounded-xl bg-cyan-600 px-4 py-2 font-semibold text-white hover:bg-cyan-700 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                Download Image
              </button>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Delete Augmented Result?"
        message={`This will delete '${deleteTarget?.output_filename}' from the outputs folder and Results Gallery. Your original uploaded image ('${deleteTarget?.original_filename}') will NOT be deleted.`}
        confirmLabel="Delete Result"
        loading={deleting}
        onConfirm={handleConfirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
