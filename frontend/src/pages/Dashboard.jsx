import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  UploadCloud,
  Wand2,
  Layers,
  Images,
  ImagePlus,
  Sparkles,
  Cpu,
  Sliders,
  ArrowRight,
  RefreshCw,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { extractErrorMessage, getDashboard, resolveMediaUrl } from '../services/api';
import {
  EmptyState,
  ErrorBanner,
  LoadingSpinner,
  StatCard,
  StatusBadge,
  formatBytes,
  formatDateTime,
} from '../components/UiStates';

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchDashboard = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getDashboard();
      setData(res);
    } catch (err) {
      setError(extractErrorMessage(err, 'Failed to load dashboard metrics.'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  if (loading) return <LoadingSpinner label="Loading ReefAugment AI dashboard..." />;
  if (error) return <ErrorBanner message={error} onRetry={fetchDashboard} />;
  if (!data) return null;

  const hasDataset = data.total_uploaded_images > 0 || data.total_augmented_images > 0;

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="rounded-2xl bg-gradient-to-r from-[#041428] via-[#092b4c] to-[#0c4a6e] p-6 sm:p-7 text-white shadow-lg">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-5">
          <div className="max-w-2xl">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-400/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-300/30">
              Underwater Computer Vision Workbench
            </span>
            <h2 className="text-xl sm:text-2xl font-bold mt-3 tracking-tight">
              Expand & Balance Your Underwater Species Training Datasets
            </h2>
            <p className="text-sm text-cyan-100/80 mt-1.5 leading-relaxed">
              Simulate water-column haze, marine snow particles, lighting shifts, and geometric diversity locally with OpenCV and Albumentations.
            </p>
          </div>
          <div className="flex flex-wrap gap-2.5">
            <Link
              to="/upload"
              className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-4 py-2.5 text-xs font-bold text-slate-950 hover:bg-cyan-300 transition shadow-sm"
            >
              <UploadCloud className="w-4 h-4" />
              Upload Image
            </Link>
            <Link
              to="/studio"
              className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2.5 text-xs font-semibold text-white ring-1 ring-white/20 hover:bg-white/20 transition"
            >
              <Wand2 className="w-4 h-4 text-cyan-300" />
              Open Studio
            </Link>
            <Link
              to="/generate"
              className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2.5 text-xs font-semibold text-white ring-1 ring-white/20 hover:bg-white/20 transition"
            >
              <Sparkles className="w-4 h-4 text-teal-300" />
              AI Generation
            </Link>
            <Link
              to="/batch"
              className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2.5 text-xs font-semibold text-white ring-1 ring-white/20 hover:bg-white/20 transition"
            >
              <Layers className="w-4 h-4 text-teal-300" />
              Batch Process
            </Link>
            <Link
              to="/gallery"
              className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2.5 text-xs font-semibold text-white ring-1 ring-white/20 hover:bg-white/20 transition"
            >
              <Images className="w-4 h-4 text-cyan-200" />
              View Results
            </Link>
          </div>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          title="Total Uploaded Images"
          value={data.total_uploaded_images}
          subtitle="Original preserved species photos"
          icon={ImagePlus}
          accent="cyan"
        />
        <StatCard
          title="Total Augmented Images"
          value={data.total_augmented_images}
          subtitle={`Total storage: ${formatBytes(data.total_storage_bytes)}`}
          icon={Sparkles}
          accent="teal"
        />
        <StatCard
          title="Total Processing Jobs"
          value={data.total_processing_jobs}
          subtitle="Single, pipeline & batch executions"
          icon={Cpu}
          accent="indigo"
        />
        <StatCard
          title="Augmentation Techniques"
          value={data.total_techniques}
          subtitle="Geometric, color, optics & underwater"
          icon={Sliders}
          accent="amber"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Augmentation Activity Timeline</h3>
              <p className="text-xs text-slate-500">Daily uploads & generated images from SQLite</p>
            </div>
            <button
              type="button"
              onClick={fetchDashboard}
              className="p-2 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 cursor-pointer"
              title="Refresh dashboard"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          {data.activity_chart.length === 0 ? (
            <EmptyState
              title="No activity recorded yet"
              description="Upload underwater species images and run augmentations to see daily activity trends."
            />
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.activity_chart}>
                  <defs>
                    <linearGradient id="augGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0891b2" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#0891b2" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="upGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#64748b' }} />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="uploaded"
                    name="Uploaded"
                    stroke="#14b8a6"
                    fill="url(#upGrad)"
                    strokeWidth={2}
                  />
                  <Area
                    type="monotone"
                    dataKey="augmented"
                    name="Augmented"
                    stroke="#0891b2"
                    fill="url(#augGrad)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80">
          <div className="mb-4">
            <h3 className="text-sm font-bold text-slate-900">Dataset Technique Breakdown</h3>
            <p className="text-xs text-slate-500">Generated outputs grouped by augmentation method</p>
          </div>
          {data.technique_chart.length === 0 ? (
            <EmptyState
              title="No augmented outputs yet"
              description="Run single or batch augmentations to populate technique usage analytics."
            />
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.technique_chart} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11, fill: '#64748b' }} />
                  <YAxis
                    type="category"
                    dataKey="label"
                    width={130}
                    tick={{ fontSize: 11, fill: '#334155' }}
                  />
                  <Tooltip />
                  <Bar dataKey="count" name="Generated Images" fill="#0891b2" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Recent Uploads & Generated Images */}
      {!hasDataset ? (
        <EmptyState
          title="Your ReefAugment AI workspace is ready"
          description="No underwater images have been uploaded yet. Start by uploading marine species photos (JPG, JPEG, PNG, or WEBP)."
          action={
            <Link
              to="/upload"
              className="inline-flex items-center gap-2 rounded-xl bg-cyan-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-cyan-700 transition"
            >
              <UploadCloud className="w-4 h-4" />
              Upload First Images
            </Link>
          }
        />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Uploaded */}
          <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-slate-900">Recent Uploaded Images</h3>
              <Link to="/upload" className="inline-flex items-center gap-1 text-xs font-semibold text-cyan-700 hover:text-cyan-800">
                Manage <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
            {data.recent_uploaded_images.length === 0 ? (
              <p className="text-xs text-slate-500 py-6 text-center">No uploaded images yet.</p>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {data.recent_uploaded_images.map((img) => (
                  <Link
                    key={img.id}
                    to={`/studio?imageId=${img.id}`}
                    className="group rounded-xl border border-slate-200 overflow-hidden bg-slate-50 hover:border-cyan-400 transition"
                  >
                    <div className="aspect-video bg-slate-200 overflow-hidden">
                      <img
                        src={resolveMediaUrl(img.image_url)}
                        alt={img.original_filename}
                        className="w-full h-full object-cover group-hover:scale-105 transition duration-200"
                      />
                    </div>
                    <div className="p-2">
                      <p className="text-xs font-semibold text-slate-800 truncate">{img.original_filename}</p>
                      <p className="text-[11px] text-slate-500">
                        {img.width}x{img.height} • {formatBytes(img.file_size)}
                      </p>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>

          {/* Recent Generated */}
          <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-slate-900">Recent Generated Images</h3>
              <Link to="/gallery" className="inline-flex items-center gap-1 text-xs font-semibold text-cyan-700 hover:text-cyan-800">
                Gallery <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
            {data.recent_generated_images.length === 0 ? (
              <p className="text-xs text-slate-500 py-6 text-center">No generated augmentations yet.</p>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {data.recent_generated_images.map((res) => (
                  <Link
                    key={res.id}
                    to={`/comparison?originalId=${res.original_image_id}&resultId=${res.id}`}
                    className="group rounded-xl border border-slate-200 overflow-hidden bg-slate-50 hover:border-teal-400 transition"
                  >
                    <div className="aspect-video bg-slate-200 overflow-hidden">
                      <img
                        src={resolveMediaUrl(res.image_url)}
                        alt={res.output_filename}
                        className="w-full h-full object-cover group-hover:scale-105 transition duration-200"
                      />
                    </div>
                    <div className="p-2">
                      <p className="text-xs font-semibold text-slate-800 truncate">{res.technique_label}</p>
                      <p className="text-[11px] text-slate-500 truncate">{res.original_filename}</p>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recent Processing Activity */}
      <div className="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200/80">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Recent Processing Activity</h3>
            <p className="text-xs text-slate-500">Latest augmentation jobs recorded in SQLite</p>
          </div>
          <Link to="/history" className="inline-flex items-center gap-1 text-xs font-semibold text-cyan-700 hover:text-cyan-800">
            Full History <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
        {data.recent_activity.length === 0 ? (
          <p className="text-xs text-slate-500 py-6 text-center">No processing activity yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="py-2.5 pr-4">Job</th>
                  <th className="py-2.5 pr-4">Original Image</th>
                  <th className="py-2.5 pr-4">Technique</th>
                  <th className="py-2.5 pr-4">Status</th>
                  <th className="py-2.5">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs">
                {data.recent_activity.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/80">
                    <td className="py-2.5 pr-4 font-mono text-slate-600">#{item.job_id || '—'}</td>
                    <td className="py-2.5 pr-4 font-medium text-slate-800">{item.original_filename}</td>
                    <td className="py-2.5 pr-4 text-slate-700">{item.technique_label}</td>
                    <td className="py-2.5 pr-4">
                      <StatusBadge status={item.status} />
                    </td>
                    <td className="py-2.5 text-slate-500">{formatDateTime(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
