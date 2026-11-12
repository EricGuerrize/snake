import { defineConfig } from "vite";

// Paths relativos: funciona em Vercel (qualquer domínio) sem base path fixo.
export default defineConfig({
  base: "./",
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  test: {
    environment: "node",
  },
});
