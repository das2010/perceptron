import { describe, expect, it } from "vitest";

import css from "./tokens.css?raw";

// Paleta del Manual de Marca Preteco (SPEC §11.1).
const BRAND = {
  "--pt-lime": "#cbff00",
  "--pt-black": "#263000",
  "--pt-dark": "#334000",
  "--pt-white": "#f7ffd6",
  "--pt-light": "#faffeb",
  "--pt-neutral": "#d1d1c3",
  "--pt-violet": "#cd87ff",
  "--pt-violet-dark": "#a46dcd",
  "--pt-violet-light": "#e6c3ff",
};

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function luminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

describe("tokens de marca", () => {
  it.each(Object.entries(BRAND))("%s = %s", (token, hex) => {
    expect(css).toMatch(new RegExp(`${token}:\\s*${hex};`, "i"));
  });

  it("combinaciones principales cumplen WCAG AA (≥ 4.5:1)", () => {
    expect(contrast(BRAND["--pt-black"], BRAND["--pt-light"])).toBeGreaterThanOrEqual(4.5); // texto claro
    expect(contrast(BRAND["--pt-black"], BRAND["--pt-lime"])).toBeGreaterThanOrEqual(4.5); // botón primario
    expect(contrast(BRAND["--pt-light"], BRAND["--pt-dark"])).toBeGreaterThanOrEqual(4.5); // texto oscuro
    expect(contrast(BRAND["--pt-lime"], BRAND["--pt-dark"])).toBeGreaterThanOrEqual(4.5); // logo píldora
  });

  it("el Lima no alcanza contraste de texto sobre fondo claro (regla §11.2)", () => {
    expect(contrast(BRAND["--pt-lime"], BRAND["--pt-light"])).toBeLessThan(4.5);
  });
});
