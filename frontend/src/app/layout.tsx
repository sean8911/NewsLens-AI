import type { Metadata } from "next";
import { Shell } from "@/components/shell";
import { LocaleProvider } from "@/components/locale";
import "./globals.css";
import "./metro.css";

export const metadata: Metadata = {
  title: {
    default: "NewsLens AI — A clearer view",
    template: "%s | NewsLens AI",
  },
  description:
    "Explore the news, connect coverage, and ask questions grounded in original reporting.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <LocaleProvider>
          <Shell>{children}</Shell>
        </LocaleProvider>
      </body>
    </html>
  );
}
