import { useEffect, useRef } from "react";

let firstPage = true;

/**
 * Смена экрана для клавиатуры и скринридера: заголовок вкладки, прокрутка
 * наверх (или к #якорю из ссылки) и фокус на h1 страницы. На первой загрузке
 * фокус не трогаем — он и так в начале документа.
 */
export function usePageFocus<T extends HTMLElement>(title: string) {
  const ref = useRef<T>(null);
  useEffect(() => {
    document.title = `${title} — ReID ТС`;
    const hash = window.location.hash.slice(1);
    const target = hash ? document.getElementById(decodeURIComponent(hash)) : null;
    if (target) {
      target.scrollIntoView();
    } else if (!firstPage) {
      window.scrollTo(0, 0);
    }
    if (!firstPage) ref.current?.focus({ preventScroll: true });
    firstPage = false;
  }, [title]);
  return ref;
}
