import type { NextConfig } from "next";

const config: NextConfig = {
  output: process.env.NEWSLENS_STANDALONE === "1" ? "standalone" : undefined,
  poweredByHeader: false,
  devIndicators: false,
};
export default config;
