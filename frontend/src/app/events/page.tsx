import { Suspense } from "react";
import { Events } from "@/components/browse";
import { Loading } from "@/components/ui";
export const metadata = { title: "News events" };
export default function Page() {
  return (
    <Suspense fallback={<Loading />}>
      <Events />
    </Suspense>
  );
}
