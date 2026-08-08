import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/nav";

export const metadata: Metadata = {
  title: "FinAscend — cash health",
  description:
    "Plain-language cash-flow guidance for small businesses, with the working shown behind every number.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="topbar">
            <div className="topbar-inner">
              <a className="wordmark" href="/">
                <span className="wordmark-dot" aria-hidden="true" />
                FinAscend
              </a>
              <Nav />
            </div>
          </header>
          <main className="main">{children}</main>
          <footer className="footer">
            <div className="footer-inner">
              Every figure on this site is fetched live from the FinAscend API when the
              page loads. Nothing is stored in the page, and no number is estimated in
              the browser — if a value cannot be fetched, the page says so instead of
              showing a placeholder.
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
