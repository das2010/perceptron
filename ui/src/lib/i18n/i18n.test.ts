import { describe, expect, it } from "vitest";

import en from "./en.json";
import es from "./es.json";

function keys(obj: object, prefix = ""): string[] {
  return Object.entries(obj).flatMap(([k, v]) =>
    typeof v === "object" && v !== null ? keys(v as object, `${prefix}${k}.`) : [`${prefix}${k}`],
  );
}

describe("i18n", () => {
  it("es y en tienen exactamente las mismas claves", () => {
    expect(keys(en).sort()).toEqual(keys(es).sort());
  });

  it("no hay traducciones vacías", () => {
    for (const dict of [es, en]) {
      for (const key of keys(dict)) {
        const value = key
          .split(".")
          .reduce<unknown>((o, k) => (o as Record<string, unknown>)[k], dict);
        expect(value, key).not.toBe("");
      }
    }
  });
});
