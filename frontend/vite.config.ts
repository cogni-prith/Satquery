import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // The backend runs on 8000. Proxying keeps the frontend origin-clean in dev and means
    // the deployed build can be served from the same origin with no code change.
    proxy: { '/api': 'http://localhost:8000' },
  },
})
