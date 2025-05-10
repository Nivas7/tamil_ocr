import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const Sans = localFont({
  src: "../../public/fonts/PPNeueMontreal-Regular.ttf",
  variable: "--Sans",
});

const Serif = localFont({
  src: "../../public/fonts/PPNeueWorld-Regular.otf",
  variable: "--Serif",
});

export const metadata: Metadata = {
  title: "Tamil Ocr",
  description:
    "Tamil Ocr that is used to recognize the tamil hnad Written Charters",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${Sans.variable} ${Serif.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
