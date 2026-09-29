import { BookOpenText, Braces, Images, Moon, ScanSearch, Sun } from "lucide-react";
import type { Info } from "../api/types";
import { Link, ROUTES, type Route } from "./router";
import type { Theme } from "./theme";
import type { HealthState } from "./useHealth";

const NAV: { route: Route; label: string; icon: typeof ScanSearch }[] = [
  { route: "search", label: "Поиск", icon: ScanSearch },
  { route: "gallery", label: "Галерея", icon: Images },
  { route: "solution", label: "Решение по ТЗ", icon: BookOpenText },
];

export function Header(props: {
  route: Route;
  health: HealthState;
  info: Info | null;
  theme: Theme;
  onToggleTheme: () => void;
}) {
  return (
    <header className="header">
      <div className="header__inner">
        <Link href={ROUTES.search} className="brand" aria-label="ReID ТС — на экран поиска">
          <svg className="brand__mark" viewBox="0 0 32 32" aria-hidden="true">
            <path
              d="M7 12V7h5M20 7h5v5M25 20v5h-5M12 25H7v-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <circle cx="16" cy="16" r="3.2" className="brand__dot" />
          </svg>
          <span className="brand__text">
            <span className="brand__name">ReID ТС</span>
            <span className="brand__tag">поиск автомобиля без ГРЗ</span>
          </span>
        </Link>

        <nav className="nav" aria-label="Разделы">
          {NAV.map(({ route, label, icon: Icon }) => (
            <Link
              key={route}
              href={ROUTES[route]}
              className="nav__link"
              aria-current={props.route === route ? "page" : undefined}
            >
              <Icon aria-hidden="true" />
              <span>{label}</span>
            </Link>
          ))}
        </nav>

        <div className="header__tools">
          <Status health={props.health} info={props.info} />
          <a className="btn btn--ghost btn--sm header__api" href="/api/docs" target="_blank" rel="noreferrer">
            <Braces aria-hidden="true" />
            <span>API</span>
          </a>
          <button
            type="button"
            className="btn btn--ghost btn--sm btn--icon"
            onClick={props.onToggleTheme}
            aria-label={props.theme === "dark" ? "Светлая тема" : "Тёмная тема"}
            title={props.theme === "dark" ? "Светлая тема" : "Тёмная тема"}
          >
            {props.theme === "dark" ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
          </button>
        </div>
      </div>
    </header>
  );
}

function Status({ health, info }: { health: HealthState; info: Info | null }) {
  const model = info?.model;
  const kind = health.kind;
  const label =
    kind === "loading"
      ? "проверяем сервисы"
      : kind === "ok"
        ? "сервисы в работе"
        : kind === "degraded"
          ? `не отвечает: ${(["gallery", "search", "inference"] as const)
              .filter((s) => !health.health[s])
              .join(", ")}`
          : "API недоступен";
  return (
    <div className={`status status--${kind}`} role="status" aria-live="polite">
      <span className="status__dot" aria-hidden="true" />
      <span className="status__text">
        <span className="status__label">{label}</span>
        {model && (
          <span className="status__model mono" title={`${model.name} ${model.version}`}>
            {shortModel(model.name)} · {model.dim}-d · {model.executionProvider}
          </span>
        )}
      </span>
    </div>
  );
}

/** vitl16_dinov3_2plus2_fitv8 → DINOv3 ViT-L/16 */
function shortModel(name: string): string {
  if (/dinov3/i.test(name) && /vitl16/i.test(name)) return "DINOv3 ViT-L/16";
  return name;
}
