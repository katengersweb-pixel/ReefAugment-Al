import React, { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  UploadCloud,
  Wand2,
  Layers,
  Images,
  SplitSquareHorizontal,
  BarChart3,
  History,
  Waves,
  Sparkles,
  Menu,
  X,
  Download,
  ChevronRight,
} from 'lucide-react';
import { downloadFileFromEndpoint, getHealth } from '../services/api';
import { useToast } from '../components/ToastContext';

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/upload', label: 'Image Upload', icon: UploadCloud },
  { to: '/studio', label: 'Augmentation Studio', icon: Wand2 },
  { to: '/generate', label: 'AI Generation Studio', icon: Sparkles },
  { to: '/batch', label: 'Batch Processing', icon: Layers },
  { to: '/gallery', label: 'Results Gallery', icon: Images },
  { to: '/comparison', label: 'Before & After Comparison', icon: SplitSquareHorizontal },
  { to: '/summary', label: 'Dataset Summary', icon: BarChart3 },
  { to: '/history', label: 'Processing History', icon: History },
];

export default function MainLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [apiOnline, setApiOnline] = useState(null);
  const [exporting, setExporting] = useState(false);
  const location = useLocation();
  const toast = useToast();

  const activeItem =
    NAV_ITEMS.find((item) =>
      item.to === '/' ? location.pathname === '/' : location.pathname.startsWith(item.to)
    ) || NAV_ITEMS[0];

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        await getHealth();
        if (mounted) setApiOnline(true);
      } catch {
        if (mounted) setApiOnline(false);
      }
    };
    check();
    const timer = window.setInterval(check, 20000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  const handleQuickExportZip = async () => {
    setExporting(true);
    try {
      await downloadFileFromEndpoint('/api/export/zip', 'reefaugment_dataset_export.zip');
      toast.success('Dataset ZIP archive downloaded with manifest.json.');
    } catch (err) {
      const msg =
        err?.response?.status === 404
          ? 'No generated images available to export yet.'
          : 'Unable to export ZIP archive. Check that the backend is running.';
      toast.error(msg);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-slate-100 text-slate-900">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-950/60 backdrop-blur-xs lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-72 flex flex-col bg-gradient-to-b from-[#041428] via-[#07213e] to-[#0a2f52] text-slate-100 shadow-2xl transition-transform duration-200 lg:static lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-teal-500 flex items-center justify-center shadow-lg shadow-cyan-500/25">
              <Waves className="w-6 h-6 text-slate-950" />
            </div>
            <div>
              <span className="block text-base font-bold tracking-tight text-white">
                ReefAugment AI
              </span>
              <span className="block text-[11px] font-medium text-cyan-300/80">
                Marine Vision Dataset Lab
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden text-slate-300 hover:text-white p-1"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-3.5 py-5 space-y-1">
          <p className="px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-cyan-200/50">
            Workspace Modules
          </p>
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/25 to-teal-500/15 text-white ring-1 ring-cyan-400/40 shadow-sm'
                    : 'text-slate-300 hover:bg-white/5 hover:text-white'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0 text-cyan-300" />
              <span className="truncate">{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-white/10">
          <div className="rounded-xl bg-white/5 border border-white/10 p-3.5">
            <p className="text-xs font-semibold text-cyan-200">Local Offline Pipeline</p>
            <p className="text-[11px] text-slate-300/80 mt-1 leading-relaxed">
              Images & SQLite records are stored locally on your machine with OpenCV & Albumentations.
            </p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="sticky top-0 z-20 bg-white/90 backdrop-blur-md border-b border-slate-200/80 px-4 sm:px-8 py-3.5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden rounded-xl border border-slate-200 p-2 text-slate-600 hover:bg-slate-100"
                aria-label="Open navigation"
              >
                <Menu className="w-5 h-5" />
              </button>
              <div>
                <div className="flex items-center gap-1.5 text-xs text-slate-400">
                  <span>ReefAugment AI</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                  <span className="text-cyan-700 font-medium">{activeItem.label}</span>
                </div>
                <h1 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight">
                  {activeItem.label}
                </h1>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold border ${
                  apiOnline === true
                    ? 'bg-teal-50 text-teal-700 border-teal-200'
                    : apiOnline === false
                    ? 'bg-rose-50 text-rose-700 border-rose-200'
                    : 'bg-slate-100 text-slate-600 border-slate-200'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    apiOnline === true
                      ? 'bg-teal-500 animate-pulse'
                      : apiOnline === false
                      ? 'bg-rose-500'
                      : 'bg-slate-400'
                  }`}
                />
                {apiOnline === true
                  ? 'Backend Online'
                  : apiOnline === false
                  ? 'Backend Offline'
                  : 'Checking API...'}
              </div>

              <button
                type="button"
                onClick={handleQuickExportZip}
                disabled={exporting}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#07213e] to-[#0a4b6e] px-3.5 py-2 text-xs font-semibold text-white shadow-xs hover:from-[#0a2d52] hover:to-[#0e5f8a] transition cursor-pointer disabled:opacity-60"
              >
                <Download className="w-4 h-4 text-cyan-300" />
                <span className="hidden sm:inline">
                  {exporting ? 'Preparing ZIP...' : 'Export Dataset ZIP'}
                </span>
              </button>
            </div>
          </div>
        </header>

        <main className="flex-1 p-4 sm:p-8 max-w-7xl w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
