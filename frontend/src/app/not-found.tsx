import Link from "next/link";
import { Empty } from "@/components/ui";
export default function NotFound() {
  return (
    <Empty
      title="This page isn’t in the picture."
      action={
        <Link className="button" href="/">
          Back to overview
        </Link>
      }
    >
      Find a new perspective in your news library.
    </Empty>
  );
}
