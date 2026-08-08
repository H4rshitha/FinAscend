import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/ui";

export const metadata: Metadata = {
  title: "FinAscend — liquidity terminal",
  description:
    "Runway-at-Risk, solver comparison, credit risk and receipt ingestion, every figure served live by the FinAscend API.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* Two families, loaded together: Inter for prose, JetBrains Mono for
            every number in the product. Tabular figures are the reason for the
            second family — a column of amounts must align on its decimal. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="shell">
          <Sidebar />
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
