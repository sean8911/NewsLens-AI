import { Suspense } from "react";
import { Trending } from "@/components/browse";
import { Loading } from "@/components/ui";
export const metadata = { title: "Trending topics" };
export default function Page() {
  return (
    <Suspense fallback={<Loading />}>
      <Trending />
    </Suspense>
  );
}
