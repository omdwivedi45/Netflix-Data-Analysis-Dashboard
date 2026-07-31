import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Netflix Data Analyst Hub | 10/10 Portfolio Project",
  description: "Executive Data Analyst Portfolio showcasing Python ETL, Normalized SQLite Engine, Live SQL Runner, Automated Excel Reports, and Power BI / Tableau Analytics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full dark">
      <body className="min-h-full flex flex-col bg-[#0E1117] text-[#E5E5E5] selection:bg-[#E50914] selection:text-white">
        {children}
      </body>
    </html>
  );
}
