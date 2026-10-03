import { defineConfig } from "@playwright/test";

// These regression tests use Playwright's HTTP client, not a browser profile.
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  reporter: [
    ["list"],
    [
      "json",
      {
        outputFile:
          process.env.NEWSLENS_TEST_REPORT || "../docs/frontend-tests.json",
      },
    ],
  ],
  use: { baseURL: "http://127.0.0.1:3001" },
  webServer: {
    command:
      "node node_modules/next/dist/bin/next start --hostname 127.0.0.1 --port 3001",
    url: "http://127.0.0.1:3001",
    reuseExistingServer: false,
    env: {
      NEWSLENS_API_URL: `http://127.0.0.1:${process.env.NEWSLENS_TEST_BACKEND_PORT || "8010"}`,
    },
    timeout: 30_000,
  },
});
