/**
 * Componentes base estilo shadcn/ui (Radix + Tailwind) tematizados con los tokens Preteco.
 * Regla §11.2: el botón primario es fondo Lima + texto Negro en ambos temas; todo lo que
 * viene del LLM usa la familia "copilot" (violeta) y el ícono de destellos.
 */
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Slot } from "@radix-ui/react-slot";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cva, type VariantProps } from "class-variance-authority";
import { Loader2, Sparkles, X } from "lucide-react";
import type {
  ComponentProps,
  HTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
} from "react";
import { useTranslation } from "react-i18next";

import { cn } from "@/lib/cn";

// ---------------------------------------------------------------- botón

const button = cva(
  "inline-flex items-center justify-center gap-2 rounded-pt text-sm font-semibold transition-colors disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-primary text-on-primary hover:brightness-95",
        secondary: "border border-line bg-card text-ink hover:bg-canvas",
        ghost: "text-ink hover:bg-card",
        danger: "bg-bad text-canvas hover:brightness-95",
        ai: "bg-copilot-bg text-ink border border-copilot hover:brightness-95",
      },
      size: { sm: "h-8 px-3", md: "h-10 px-4", icon: "h-9 w-9" },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps extends ComponentProps<"button">, VariantProps<typeof button> {
  asChild?: boolean;
  loading?: boolean;
}

export function Button({
  className,
  variant,
  size,
  asChild,
  loading,
  children,
  disabled,
  ...props
}: ButtonProps) {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp
      className={cn(button({ variant, size }), className)}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
      {children}
    </Comp>
  );
}

// ---------------------------------------------------------------- superficies

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("rounded-pt border border-line bg-card p-5 shadow-sm", className)}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return <h2 className={cn("mb-3 text-base font-semibold", className)} {...props} />;
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold">{title}</h1>
        {description && <p className="mt-1 text-sm text-muted">{description}</p>}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}

// ---------------------------------------------------------------- estados

const badge = cva("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold", {
  variants: {
    tone: {
      neutral: "bg-canvas text-muted border border-line",
      ok: "bg-canvas text-ok border border-ok",
      warn: "bg-canvas text-warn border border-warn",
      bad: "bg-canvas text-bad border border-bad",
      brand: "bg-pt-dark text-pt-lime",
    },
  },
  defaultVariants: { tone: "neutral" },
});

export function Badge({
  tone,
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badge>) {
  return <span className={cn(badge({ tone }), className)} {...props} />;
}

/** Marca de "sugerido por IA" (SPEC §11.2): violeta + ícono. */
export function AiBadge({ className }: { className?: string }) {
  const { t } = useTranslation();
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-copilot bg-copilot-bg px-2 py-0.5 text-xs font-semibold text-ink",
        className,
      )}
    >
      <Sparkles className="h-3 w-3 text-copilot" aria-hidden="true" />
      {t("ai.suggested")}
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  const { t } = useTranslation();
  return (
    <p className="flex items-center gap-2 text-sm text-muted" role="status">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
      {label ?? t("common.loading")}
    </p>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const { t } = useTranslation();
  if (!error) return null;
  const status = (error as { status?: number }).status;
  const offline = error instanceof TypeError || status === 502 || status === 503 || status === 504;
  const message = offline
    ? t("engine.error")
    : error instanceof Error
      ? error.message
      : t("common.error");
  return (
    <p role="alert" className="rounded-pt border border-bad p-3 text-sm text-bad">
      {message}
    </p>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-pt border border-dashed border-line p-6 text-center text-sm text-muted">
      {children}
    </p>
  );
}

// ---------------------------------------------------------------- formularios

export function Field({
  label,
  hint,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="font-semibold">{label}</span>
      {children}
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </label>
  );
}

const control =
  "h-10 rounded-pt border border-line bg-canvas px-3 text-sm text-ink placeholder:text-muted";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(control, className)} {...props} />;
}

export function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cn(control, className)} {...props} />;
}

export function Textarea({ className, ...props }: ComponentProps<"textarea">) {
  return <textarea className={cn(control, "h-auto min-h-20 py-2", className)} {...props} />;
}

// ---------------------------------------------------------------- tablas

export function Table({ className, ...props }: ComponentProps<"table">) {
  return (
    <div className="overflow-x-auto">
      <table className={cn("w-full border-collapse text-sm", className)} {...props} />
    </div>
  );
}

export function Th({ className, ...props }: ComponentProps<"th">) {
  return (
    <th
      className={cn(
        "border-b border-line px-3 py-2 text-left text-xs font-semibold uppercase text-muted",
        className,
      )}
      {...props}
    />
  );
}

export function Td({ className, ...props }: ComponentProps<"td">) {
  return <td className={cn("border-b border-line px-3 py-2 align-top", className)} {...props} />;
}

// ---------------------------------------------------------------- pestañas y diálogos

export const Tabs = TabsPrimitive.Root;

export function TabsList({ className, ...props }: ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn("mb-4 flex gap-1 border-b border-line", className)}
      {...props}
    />
  );
}

export function TabsTrigger({ className, ...props }: ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        "border-b-2 border-transparent px-3 py-2 text-sm text-muted data-[state=active]:border-brand data-[state=active]:font-semibold data-[state=active]:text-ink",
        className,
      )}
      {...props}
    />
  );
}

export const TabsContent = TabsPrimitive.Content;

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 bg-black/40" />
        <DialogPrimitive.Content className="fixed left-1/2 top-1/2 w-[min(92vw,32rem)] -translate-x-1/2 -translate-y-1/2 rounded-pt border border-line bg-card p-6 text-ink shadow-lg">
          <DialogPrimitive.Title className="text-lg font-semibold">{title}</DialogPrimitive.Title>
          {description ? (
            <DialogPrimitive.Description className="mt-1 text-sm text-muted">
              {description}
            </DialogPrimitive.Description>
          ) : (
            <DialogPrimitive.Description className="sr-only">{title}</DialogPrimitive.Description>
          )}
          <div className="mt-4">{children}</div>
          <DialogPrimitive.Close
            className="absolute right-3 top-3 rounded-pt p-1 text-muted hover:text-ink"
            aria-label={t("common.close")}
          >
            <X className="h-4 w-4" />
          </DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

/** Tarjeta de propuesta del LLM: siempre aceptable/rechazable (CLAUDE.md, §11.2). */
export function AiSuggestion({
  title,
  children,
  onAccept,
  onReject,
  accepted,
}: {
  title: ReactNode;
  children: ReactNode;
  onAccept?: () => void;
  onReject?: () => void;
  accepted?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <div className="rounded-pt border border-copilot bg-copilot-bg/40 p-4">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="font-semibold">{title}</h3>
        <AiBadge />
      </div>
      <div className="text-sm">{children}</div>
      {(onAccept || onReject) && (
        <div className="mt-3 flex gap-2">
          {onAccept && (
            <Button size="sm" variant={accepted ? "primary" : "ai"} onClick={onAccept}>
              {accepted ? t("ai.accepted") : t("ai.accept")}
            </Button>
          )}
          {onReject && (
            <Button size="sm" variant="ghost" onClick={onReject}>
              {t("ai.reject")}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
