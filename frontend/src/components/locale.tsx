/** @jsxImportSource react */
"use client";

import { createContext, useContext, useEffect, useReducer } from "react";
import { date } from "@/lib/api";

export type Locale = "en" | "zh";
export const localeKey = "newslens-ui-language";
export const LocaleContext = createContext<{
  locale: Locale;
  setLocale: (locale: Locale) => void;
}>({ locale: "en", setLocale: () => {} });

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, dispatch] = useReducer(
    (_: Locale, next: Locale) => next,
    "en",
  );
  useEffect(() => {
    try {
      const stored = localStorage.getItem(localeKey);
      if (stored === "zh" || stored === "en") {
        dispatch(stored);
        document.documentElement.lang = stored === "zh" ? "zh-Hans" : "en";
      }
    } catch {
      // Language switching remains available without browser storage.
    }
  }, []);
  function setLocale(next: Locale) {
    dispatch(next);
    document.documentElement.lang = next === "zh" ? "zh-Hans" : "en";
    try {
      localStorage.setItem(localeKey, next);
    } catch {
      // Retain the preference in memory for this visit.
    }
  }
  return (
    <LocaleContext.Provider value={{ locale, setLocale }}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  const context = useContext(LocaleContext);
  return {
    ...context,
    t: (english: string, chinese: string) =>
      context.locale === "zh" ? chinese : english,
    formatDate: (value: string | null, withTime = false) =>
      date(value, withTime, context.locale),
  };
}
