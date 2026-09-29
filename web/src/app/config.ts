// Рантайм-конфиг: /config.js пишет entrypoint nginx из окружения контейнера
// (web/docker-entrypoint.d/50-config.sh), в dev — пустой web/public/config.js.
declare global {
  interface Window {
    __APP_CONFIG__?: { ymapsKey?: string };
  }
}

export const YMAPS_KEY: string = window.__APP_CONFIG__?.ymapsKey || import.meta.env.VITE_YMAPS_KEY || "";
