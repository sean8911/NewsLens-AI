import { EventDetail } from "@/components/event-intelligence";
export const metadata = { title: "Event intelligence" };
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <EventDetail key={id} id={id} />;
}
