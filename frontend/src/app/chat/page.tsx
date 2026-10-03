import { Suspense } from "react";
import { Chat } from "@/components/chat";
import { Loading } from "@/components/ui";
export const metadata = { title: "AI chat" };
export default function Page() {
  return (
    <Suspense fallback={<Loading />}>
      <Chat />
    </Suspense>
  );
}
