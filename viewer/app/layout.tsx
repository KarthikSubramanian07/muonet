import type { Metadata, Viewport } from "next";
import "./globals.css";

const SITE_URL = "https://muonet.pages.dev";
const TITLE = "Muonet — see inside anything dense with cosmic rays";
const DESCRIPTION =
  "Muonet is an open-source muon scattering tomography toolkit. Simulate cosmic muons passing through dense objects, reconstruct the 3D interior, and drag a slider to watch a blurry blob resolve into a tungsten cube. Real detector physics, honest reproducible benchmarks.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: TITLE,
  description: DESCRIPTION,
  applicationName: "Muonet",
  keywords: [
    "muon tomography",
    "muography",
    "cosmic ray tomography",
    "muon scattering",
    "3D reconstruction",
    "nuclear security",
    "PoCA",
    "machine learning physics",
    "voxel reconstruction",
    "open source",
  ],
  authors: [{ name: "Karthik Subramanian" }],
  creator: "Karthik Subramanian",
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: SITE_URL,
    title: TITLE,
    description: DESCRIPTION,
    siteName: "Muonet",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "Muonet reconstruction viewer" }],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: ["/og.png"],
  },
  robots: { index: true, follow: true },
  category: "science",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0a0a12" },
    { media: "(prefers-color-scheme: light)", color: "#f5f5f7" },
  ],
};

const JSON_LD = {
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  name: "Muonet",
  description: DESCRIPTION,
  codeRepository: "https://github.com/KarthikSubramanian07/muonet",
  programmingLanguage: ["Python", "TypeScript"],
  license: "https://opensource.org/licenses/MIT",
  author: { "@type": "Person", name: "Karthik Subramanian" },
  keywords: "muon tomography, muography, cosmic ray, 3D reconstruction, machine learning",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }}
        />
        {/* Set the theme before paint to avoid a flash of the wrong colors. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem('muonet-theme');if(t==='light'||t==='dark'){document.documentElement.dataset.theme=t;}}catch(e){}`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
