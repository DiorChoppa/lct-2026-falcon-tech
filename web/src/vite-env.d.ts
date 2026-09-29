/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Ключ Яндекс Карт для `npm run dev` (web/.env.local); в контейнере — YMAPS_KEY. */
  readonly VITE_YMAPS_KEY?: string;
}
