import React from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { ToastProvider } from './components/ToastContext';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import AugmentationStudio from './pages/AugmentationStudio';
import BatchProcessing from './pages/BatchProcessing';
import ResultsGallery from './pages/ResultsGallery';
import Comparison from './pages/Comparison';
import DatasetSummary from './pages/DatasetSummary';
import History from './pages/History';
import AIGenerationStudio from './pages/AIGenerationStudio';

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter basename={import.meta.env.BASE_URL}>
        <Routes>
          <Route element={<MainLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/studio" element={<AugmentationStudio />} />
            <Route path="/generate" element={<AIGenerationStudio />} />
            <Route path="/batch" element={<BatchProcessing />} />
            <Route path="/gallery" element={<ResultsGallery />} />
            <Route path="/comparison" element={<Comparison />} />
            <Route path="/summary" element={<DatasetSummary />} />
            <Route path="/history" element={<History />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  );
}
