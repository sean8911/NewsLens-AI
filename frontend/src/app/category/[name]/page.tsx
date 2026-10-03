import { Suspense } from "react";
import { Feed } from "@/components/browse";
import { Loading } from "@/components/ui";
export const metadata = { title: "Topic collection" };
export default async function Page({
  params,
}: {
  params: Promise<{ name: string }>;
}) {
  const { name } = await params;
  return (
    <Suspense fallback={<Loading />}>
      <Feed category={name} />
    </Suspense>
  );
}
