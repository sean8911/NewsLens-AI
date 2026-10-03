export const overdueMs = 30 * 60 * 1000;
export function timestamp(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}
export function relativeTime(
  value: string | null | undefined,
  now: number,
  locale = "en",
): string {
  const zh = locale === "zh";
  const parsed = timestamp(value);
  if (parsed === null || !Number.isFinite(now))
    return zh ? "暂无记录" : "Not recorded";
  const elapsed = now - parsed;
  if (elapsed < 0)
    return zh
      ? "时间晚于当前时间，请检查时钟"
      : "Future timestamp — check clock";
  if (elapsed < 60_000) return zh ? "不到一分钟前" : "Less than a minute ago";
  if (elapsed < 3_600_000) {
    const n = Math.floor(elapsed / 60_000);
    return zh ? `${n} 分钟前` : `${n} minute${n === 1 ? "" : "s"} ago`;
  }
  if (elapsed < 86_400_000) {
    const n = Math.floor(elapsed / 3_600_000);
    return zh ? `${n} 小时前` : `${n} hour${n === 1 ? "" : "s"} ago`;
  }
  const n = Math.floor(elapsed / 86_400_000);
  return zh ? `${n} 天前` : `${n} day${n === 1 ? "" : "s"} ago`;
}
export function updateLabel(
  value: string | null | undefined,
  now: number,
  locale = "en",
): string {
  const zh = locale === "zh";
  const parsed = timestamp(value);
  if (parsed === null)
    return zh ? "暂无成功更新记录" : "No successful update recorded";
  if (parsed > now) return zh ? "请检查采集时间" : "Check collection clock";
  return now - parsed >= overdueMs
    ? zh
      ? "新闻待更新"
      : "Update overdue"
    : zh
      ? "近期已检查新闻更新"
      : "Collection recently checked";
}
