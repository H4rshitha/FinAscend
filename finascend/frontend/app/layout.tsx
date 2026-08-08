import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { AuthGuard } from "@/components/guard";
import { Shell } from "@/components/shell";

export const metadata: Metadata = {
  title: "FinAscend — cash health",
  description:
    "Plain-language cash-flow guidance for small businesses, with the working shown behind every number.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* Order matters: the provider owns the session, the shell decides what
            chrome to draw around it, and the guard decides whether the page
            behind that chrome may render at all. */}
        <AuthProvider>
          <Shell>
            <AuthGuard>{children}</AuthGuard>
          </Shell>
        </AuthProvider>
      </body>
    </html>
  );
}
