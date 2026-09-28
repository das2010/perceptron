import { test } from "@playwright/test";

/** Errores del navegador y texto de la página al fallar: el log del CI alcanza para diagnosticar. */
export function withDiagnostics() {
  test.beforeEach(({ page }) => {
    page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}\n${e.stack ?? ""}`));
    page.on("console", (m) => {
      if (m.type() === "error") console.log(`[console] ${m.text()}`);
    });
  });

  test.afterEach(async ({ page }, info) => {
    if (info.status !== info.expectedStatus) {
      console.log(`[url] ${page.url()}`);
      const text = await page
        .locator("body")
        .innerText()
        .catch(() => "");
      console.log(`[main] ${text.slice(0, 3000)}`);
    }
  });
}
