"use client";
import { ErrorState } from "@/components/ui";
export default function ErrorPage({ reset }: { reset: () => void }) {
  return (
    <ErrorState
      message="Something interrupted this page. Please try again."
      retry={reset}
    />
  );
}
