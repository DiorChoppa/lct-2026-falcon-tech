import { lazy, Suspense } from "react";
import { Header } from "./app/Header";
import { routeOf, routeVehicle, usePath } from "./app/router";
import { useTheme } from "./app/theme";
import { useHealth, useInfo } from "./app/useHealth";
import { SearchPage } from "./pages/search/SearchPage";
import { useSearchSession } from "./pages/search/useSearchSession";

// Галерея и описание решения грузятся отдельными чанками: оператору на
// первом экране нужен только поиск.
const GalleryPage = lazy(() => import("./pages/gallery/GalleryPage"));
const SolutionPage = lazy(() => import("./pages/solution/SolutionPage"));
const RoutePage = lazy(() => import("./pages/route/RoutePage"));

export default function App() {
  const path = usePath();
  const route = routeOf(path);
  const [theme, toggleTheme] = useTheme();
  const health = useHealth();
  const info = useInfo();
  // Состояние поиска живёт выше экранов: ушли в «Решение» и вернулись — кадр, bbox и кандидаты на месте.
  const session = useSearchSession();

  return (
    <>
      <a className="skip-link" href="#main">
        К содержимому
      </a>
      <Header route={route} health={health} info={info} theme={theme} onToggleTheme={toggleTheme} />
      <main id="main" className="main">
        <Suspense fallback={<PageFallback />}>
          {route === "search" && <SearchPage session={session} info={info} />}
          {route === "gallery" && <GalleryPage />}
          {route === "solution" && <SolutionPage info={info} />}
          {route === "route" && <RoutePage vehicleId={routeVehicle(path)} session={session} theme={theme} />}
        </Suspense>
      </main>
      <footer className="footer">
        <span>ЛЦТ 2026 · кейс Street Falcon · ГРЗ в пайплайне не используется</span>
        <span className="mono">{info?.model ? `${info.model.name} ${info.model.version}` : ""}</span>
      </footer>
    </>
  );
}

function PageFallback() {
  return (
    <div className="page" aria-busy="true">
      <div className="skeleton" style={{ height: 40, width: 280 }} />
      <div className="skeleton" style={{ height: 320, marginTop: 24 }} />
    </div>
  );
}
