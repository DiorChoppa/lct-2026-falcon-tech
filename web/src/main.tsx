import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Шрифты локально (без Google Fonts): стенд и compose работают без интернета.
import "@fontsource-variable/unbounded";
import "@fontsource-variable/manrope";
import "@fontsource-variable/jetbrains-mono";
// Базовые стили — до App: стили страниц импортируются из их модулей и должны идти после.
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/shell.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
