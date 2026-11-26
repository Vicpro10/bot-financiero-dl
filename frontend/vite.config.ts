import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// El proxy evita exponer la URL del backend; la API key NUNCA va en variables VITE_*.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
