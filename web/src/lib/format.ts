const ru = new Intl.NumberFormat("ru-RU");

/** Русское согласование: plural(3, ["кандидат", "кандидата", "кандидатов"]) → "кандидата". */
export function plural(n: number, forms: [string, string, string]): string {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return forms[0];
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return forms[1];
  return forms[2];
}

/** 0.9123 → "91 %" (узкий неразрывный пробел, как в русской типографике). */
export function pct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits).replace(".", ",")} %`;
}

/** 0.6607 → "0,661" */
export function dec(x: number, digits = 3): string {
  return x.toFixed(digits).replace(".", ",");
}

export function int(n: number): string {
  return ru.format(n);
}

/** Длительность запроса: до секунды — в мс, дальше — в секундах. */
export function duration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} мс` : `${dec(ms / 1000, 2)} с`;
}
