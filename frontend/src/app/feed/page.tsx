import { Suspense } from "react";
import { Feed } from "@/components/browse";
import { Loading } from "@/components/ui";
export const metadata = { title: "News feed" };
export default function Page() {
  return (
    <Suspense fallback={<Loading />}>
      <Feed />
    </Suspense>
  );
}
