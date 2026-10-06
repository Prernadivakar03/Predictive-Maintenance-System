import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the UI runs on :5173 and proxies API calls to the FastAPI server on :8000.
// In production `npm run build` emits frontend/dist, which FastAPI serves itself (no proxy needed).
const API_TARGET = process.env.VITE_API_PROXY || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": API_TARGET,
      "/predict": API_TARGET,
      "/health": API_TARGET,
      "/docs": API_TARGET,
      "/openapi.json": API_TARGET,
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 700 },
});
