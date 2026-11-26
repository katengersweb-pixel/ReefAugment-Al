import axios from 'axios';

const envApiUrl = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '');
const devFallbackUrl = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';

export const API_BASE_URL = envApiUrl || devFallbackUrl;
export const isApiConfigured = Boolean(API_BASE_URL);

const apiClient = axios.create({
  baseURL: API_BASE_URL || undefined,
  timeout: 120000,
});

apiClient.interceptors.request.use((config) => {
  if (!API_BASE_URL) {
    return Promise.reject(
      new Error(
        'Production API URL (VITE_API_URL) is not configured. Set VITE_API_URL in your environment or GitHub repository variables.'
      )
    );
  }
  return config;
});

export function extractErrorMessage(err, fallback = 'An unexpected error occurred.') {
  if (err?.response?.data?.detail) {
    const detail = err.response.data.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    }
    return JSON.stringify(detail);
  }
  if (err?.message) return err.message;
  return fallback;
}

export function resolveMediaUrl(relativePath) {
  if (!relativePath) return '';
  if (relativePath.startsWith('http://') || relativePath.startsWith('https://') || relativePath.startsWith('data:')) {
    return relativePath;
  }
  return `${API_BASE_URL}${relativePath.startsWith('/') ? '' : '/'}${relativePath}`;
}

export async function getHealth() {
  const { data } = await apiClient.get('/api/health');
  return data;
}

export async function getTechniques() {
  const { data } = await apiClient.get('/api/techniques');
  return data;
}

export async function getDashboard() {
  const { data } = await apiClient.get('/api/dashboard');
  return data;
}

export async function uploadImages(files, onProgress) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  const { data } = await apiClient.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(pct);
      }
    },
  });
  return data;
}

export async function getImages(search = '') {
  const params = search ? { search } : {};
  const { data } = await apiClient.get('/api/images', { params });
  return data;
}

export async function deleteImage(imageId) {
  const { data } = await apiClient.delete(`/api/images/${imageId}`);
  return data;
}

export async function augmentImage(payload) {
  const { data } = await apiClient.post('/api/augment', payload);
  return data;
}

export async function getResults(params = {}) {
  const { data } = await apiClient.get('/api/results', { params });
  return data;
}

export async function deleteResult(resultId) {
  const { data } = await apiClient.delete(`/api/results/${resultId}`);
  return data;
}

export async function startBatchJob(payload) {
  const { data } = await apiClient.post('/api/batch', payload);
  return data;
}

export async function getBatchJobs() {
  const { data } = await apiClient.get('/api/batch');
  return data;
}

export async function getBatchStatus(jobId) {
  const { data } = await apiClient.get(`/api/batch/${jobId}/status`);
  return data;
}

export async function retryBatchJob(jobId) {
  const { data } = await apiClient.post(`/api/batch/${jobId}/retry`);
  return data;
}

export async function getComparison(originalId) {
  const { data } = await apiClient.get(`/api/comparison/${originalId}`);
  return data;
}

export async function getSummary() {
  const { data } = await apiClient.get('/api/summary');
  return data;
}

export async function getSummaryCharts() {
  const { data } = await apiClient.get('/api/summary/charts');
  return data;
}

export async function getHistory(params = {}) {
  const { data } = await apiClient.get('/api/history', { params });
  return data;
}

export async function deleteHistoryRecord(recordId) {
  const { data } = await apiClient.delete(`/api/history/${recordId}`);
  return data;
}

export async function downloadFileFromEndpoint(endpointPath, fallbackFilename) {
  const response = await apiClient.get(endpointPath, { responseType: 'blob' });
  const disposition = response.headers['content-disposition'] || '';
  const match = disposition.match(/filename="?([^"]+)"?/i);
  const filename = match ? match[1] : fallbackFilename;

  const blobUrl = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = blobUrl;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(blobUrl);
}

export async function exportSelectedZip(selection) {
  const payload = Array.isArray(selection)
    ? { result_ids: selection }
    : {
        result_ids: selection?.resultIds || [],
        generated_ids: selection?.generatedIds || [],
      };
  const response = await apiClient.post('/api/export/zip', payload, {
    responseType: 'blob',
  });
  const blobUrl = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = blobUrl;
  link.setAttribute('download', 'reefaugment_selected_export.zip');
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(blobUrl);
}

// ---------------------------------------------------------------------------
// AI Image Generation Studio (Hosted SDXL)
// ---------------------------------------------------------------------------

export async function getGenerationStatus() {
  const { data } = await apiClient.get('/api/generation/status');
  return data;
}

export async function generateTextToImage(payload) {
  const { data } = await apiClient.post('/api/generation/text-to-image', payload, {
    timeout: 180000,
  });
  return data;
}

export async function generateImageToImage(formData) {
  const { data } = await apiClient.post('/api/generation/image-to-image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 180000,
  });
  return data;
}

export async function getGenerationHistory(params = {}) {
  const { data } = await apiClient.get('/api/generation/history', { params });
  return data;
}

export async function deleteGeneratedImage(genId) {
  const { data } = await apiClient.delete(`/api/generation/${genId}`);
  return data;
}

export async function sendGeneratedToAugment(genId) {
  const { data } = await apiClient.post(`/api/generation/${genId}/send-to-augment`);
  return data;
}

export default apiClient;
