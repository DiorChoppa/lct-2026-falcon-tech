import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В dev api проксируется на api-gateway (по умолчанию локальный axum; весь
// docker-стек — API_URL=http://localhost:3000), в проде это делает nginx.
const api = process.env.API_URL ?? "http://localhost:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": api, "/files": api },
  },
});
