"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowUpRight, ChevronDown } from "lucide-react";
import { FreshnessPanel, useFreshness } from "./freshness";
import { useLocale, type Locale } from "./locale";
import { updateLabel } from "@/lib/freshness";

export const categories = [
  "AI",
  "Technology",
  "Business",
  "Science",
  "Malaysia",
  "World",
];
const categoryZh: Record<string, string> = {
  AI: "人工智能",
  Technology: "科技",
  Business: "商业",
  Science: "科学",
  Malaysia: "马来西亚",
  World: "国际",
};

export function Shell({ children }: { children: React.ReactNode }) {
  const { locale, setLocale, t } = useLocale();
  const pathname = usePathname();
  const status = useFreshness();
  const isChat = pathname === "/chat";
  const mainPages = [
    {
      href: "/feed",
      label: t("news", "新闻"),
      selected:
        pathname === "/feed" ||
        pathname.startsWith("/articles/") ||
        pathname.startsWith("/category/"),
    },
    {
      href: "/search",
      label: t("search", "搜索"),
      selected: pathname === "/search",
    },
    { href: "/chat", label: t("chat", "对话"), selected: isChat },
  ];
  return (
    <div className={`metro-shell ${isChat ? "metro-chat" : ""}`}>
      <a className="skip-link" href="#main-content">
        {t("Skip to content", "跳至主要内容")}
      </a>
      <header className="metro-topbar">
        <Link className="metro-brand" href="/" aria-label="NewsLens AI">
          NewsLens <span>AI</span>
        </Link>
        <div className="metro-topbar-actions">
          <label>
            <span className="sr-only">{t("Page language", "页面语言")}</span>
            <select
              aria-label="Language / 语言"
              value={locale}
              onChange={(event) => setLocale(event.target.value as Locale)}
            >
              <option value="en">English</option>
              <option value="zh">中文</option>
            </select>
          </label>
          <span className="metro-local">
            {t("Your news workspace", "你的新闻工作区")}
          </span>
        </div>
      </header>
      <div className="metro-masthead">
        {isChat && (
          <h1>{t("explore. ask. understand.", "探索。提问。读懂。")}</h1>
        )}
        <div className="metro-nav-row">
          <nav
            className="metro-primary"
            aria-label={t("Main pages", "主要页面")}
          >
            {mainPages.map((item) => (
              <Link
                href={item.href}
                key={item.href}
                aria-current={item.selected ? "page" : undefined}
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <nav
            className="metro-secondary"
            aria-label={t("More news tools", "更多新闻工具")}
          >
            <Link href="/" aria-current={pathname === "/" ? "page" : undefined}>
              {t("Overview", "概览")}
            </Link>
            <Link
              href="/events"
              aria-current={pathname.startsWith("/events") ? "page" : undefined}
            >
              {t("Events", "新闻事件")}
            </Link>
            <Link
              href="/trending"
              aria-current={pathname === "/trending" ? "page" : undefined}
            >
              {t("Trending", "热门话题")}
            </Link>
            <details className="metro-topics">
              <summary>
                {t("Topics", "话题")} <ChevronDown size={14} />
              </summary>
              <nav aria-label={t("Categories", "分类")}>
                {categories.map((category) => (
                  <Link
                    key={category}
                    href={`/category/${category}?kind=topic`}
                    onClick={(event) =>
                      event.currentTarget
                        .closest("details")
                        ?.removeAttribute("open")
                    }
                  >
                    {locale === "zh" ? categoryZh[category] : category}
                  </Link>
                ))}
              </nav>
            </details>
          </nav>
        </div>
      </div>
      <main id="main-content" className="metro-main" tabIndex={-1}>
        {children}
        <details className="metro-coverage">
          <summary>
            <span>{t("Collection status", "新闻采集状态")}</span>
            <span>
              {status.error || status.data?.database === "unavailable"
                ? t("Status unavailable", "状态暂不可用")
                : status.data
                  ? t(
                      `${status.data.article_count} articles · ${updateLabel(status.data.last_successful_news_update, status.now, locale)}`,
                      `已收录 ${status.data.article_count} 篇 · ${updateLabel(status.data.last_successful_news_update, status.now, locale)}`,
                    )
                  : t("Checking your library…", "正在检查新闻库…")}
            </span>
            <ChevronDown size={16} />
          </summary>
          <FreshnessPanel {...status} />
        </details>
      </main>
      <footer className="metro-footer">
        <span>
          NewsLens AI <span> / </span>
          {t("Grounded in sources.", "以来源为依据。")}
        </span>
        <Link href="/feed">
          {t("Explore collected news", "浏览已收录新闻")}{" "}
          <ArrowUpRight size={14} />
        </Link>
      </footer>
    </div>
  );
}
