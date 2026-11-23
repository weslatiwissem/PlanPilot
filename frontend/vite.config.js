// Vite dev server for the PlanPilot UI.
// Every call to /api/... is forwarded to the FastAPI backend, so the browser never talks to another port
// (no CORS problems). Change API_URL if the backend runs elsewhere.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_URL = process.env.API_URL || "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_URL, changeOrigin: true, rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
});
