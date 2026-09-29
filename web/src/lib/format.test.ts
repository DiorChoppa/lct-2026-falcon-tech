import { describe, expect, it } from "vitest";
import { dec, duration, pct, plural } from "./format";

const F: [string, string, string] = ["кандидат", "кандидата", "кандидатов"];

describe("plural", () => {
  it.each([
    [1, "кандидат"],
    [2, "кандидата"],
    [4, "кандидата"],
    [5, "кандидатов"],
    [11, "кандидатов"],
    [12, "кандидатов"],
    [21, "кандидат"],
    [22, "кандидата"],
    [0, "кандидатов"],
  ])("%i → %s", (n, want) => expect(plural(n, F)).toBe(want));
});

describe("числа", () => {
  it("проценты и десятичная запятая", () => {
    expect(pct(0.9123)).toBe("91 %");
    expect(pct(0.9186, 1)).toBe("91,9 %");
    expect(dec(0.6607)).toBe("0,661");
  });
  it("длительность", () => {
    expect(duration(683.4)).toBe("683 мс");
    expect(duration(1234)).toBe("1,23 с");
  });
});
