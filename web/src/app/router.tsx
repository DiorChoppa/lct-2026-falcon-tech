import { useSyncExternalStore, type AnchorHTMLAttributes, type MouseEvent } from "react";
import { flushSync } from "react-dom";

// Три экрана — свой маленький роутер на History API вместо зависимости.
// Каждый экран доступен по прямой ссылке: nginx отдаёт index.html на любой путь.
export const ROUTES = {
  search: "/",
  gallery: "/gallery",
  solution: "/solution",
  route: "/route",
} as const;

export type Route = keyof typeof ROUTES;

const listeners = new Set<() => void>();

function subscribe(cb: () => void) {
  listeners.add(cb);
  window.addEventListener("popstate", cb);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("popstate", cb);
  };
}

export function usePath(): string {
  return useSyncExternalStore(subscribe, () => window.location.pathname);
}

export function routeOf(path: string): Route {
  if (path.startsWith(`${ROUTES.route}/`)) return "route";
  const found = (Object.keys(ROUTES) as Route[]).find((r) => ROUTES[r] === path);
  return found ?? "search";
}

/** /route/<vehicle_id> — маршрут машины (docs/superpowers/specs/2026-09-24-route-map-design.md). */
export function routeHref(vehicleId: string): string {
  return `${ROUTES.route}/${encodeURIComponent(vehicleId)}`;
}

export function routeVehicle(path: string): string | null {
  const m = /^\/route\/([^/]+)$/.exec(path);
  if (!m) return null;
  try {
    return decodeURIComponent(m[1]);
  } catch {
    return null; // битая %-последовательность в адресе, набранном руками
  }
}

const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Переход с View Transition там, где он есть; иначе просто смена DOM. */
export function navigate(to: string) {
  const url = new URL(to, window.location.href);
  if (url.pathname === window.location.pathname && url.hash === window.location.hash) return;
  const update = () => {
    window.history.pushState(null, "", url);
    listeners.forEach((l) => l());
  };
  if (!document.startViewTransition || reducedMotion() || url.pathname === window.location.pathname) {
    update();
    return;
  }
  document.startViewTransition(() => flushSync(update));
}

/** Ссылка внутри приложения: без перезагрузки, но с обычным поведением Ctrl/⌘-клика. */
export function Link({
  href,
  onClick,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) {
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    navigate(href);
  };
  return <a href={href} onClick={handle} {...rest} />;
}
