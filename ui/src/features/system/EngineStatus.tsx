import { useTranslation } from "react-i18next";

import styles from "./EngineStatus.module.css";
import { useEngineHealth } from "./useEngineHealth";

export function EngineStatus() {
  const { t } = useTranslation();
  const { data, isPending, isError, refetch } = useEngineHealth();

  let state: "pending" | "ok" | "error" = "pending";
  if (isError) state = "error";
  else if (!isPending) state = "ok";

  return (
    <section className={styles.card} aria-labelledby="engine-status-title" aria-live="polite">
      <h2 id="engine-status-title" className={styles.title}>
        {t("engine.title")}
      </h2>
      <p className={styles.status} data-state={state}>
        <span className={styles.dot} aria-hidden="true" />
        {state === "pending" && t("engine.checking")}
        {state === "ok" && t("engine.ok")}
        {state === "error" && t("engine.error")}
      </p>
      {data && <p className={styles.meta}>{t("engine.version", { version: data.version })}</p>}
      {state === "error" && (
        <button type="button" className={styles.button} onClick={() => void refetch()}>
          {t("engine.retry")}
        </button>
      )}
    </section>
  );
}
