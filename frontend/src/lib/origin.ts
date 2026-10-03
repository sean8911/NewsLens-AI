/** A configured public origin avoids relying on proxy-supplied protocol headers. */
export function allowedOrigin(
  origin: string,
  host: string | null,
  protocol: string,
  publicOrigin?: string,
): boolean {
  try {
    const parsed = new URL(origin);
    if (parsed.origin !== origin) return false;
    if (publicOrigin !== undefined) {
      const configured = new URL(publicOrigin);
      return (
        configured.protocol === "https:" &&
        configured.origin === publicOrigin &&
        parsed.origin === configured.origin
      );
    }
    return parsed.host === host && parsed.protocol === protocol;
  } catch {
    return false;
  }
}
