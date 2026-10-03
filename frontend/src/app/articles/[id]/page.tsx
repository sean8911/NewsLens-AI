import { ArticlePage } from "@/components/browse";
export const metadata = { title: "Read the story" };
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <ArticlePage id={id} />;
}
