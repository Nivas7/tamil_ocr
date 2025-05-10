"use client";
import DocumentationPage from "@/components/DocumenatationPage";
import Footer from "@/components/Footer";
import { motion, MotionValue, useScroll, useTransform } from "framer-motion";
import Lenis from "lenis";
import Error from "next/error";
import Image from "next/image";
import React, { useEffect, useRef, useState } from "react";

// Define types for props
interface SectionProps {
  scrollYProgress: MotionValue<number>;
}

interface ResponseData {
  [key: string]: string;
}

export default function Home() {
  const container = useRef<HTMLDivElement | null>(null);

  const { scrollYProgress } = useScroll({
    target: container,
    offset: ["start start", "end end"],
  });

  useEffect(() => {
    const lenis = new Lenis();

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);
  }, []);

  return (
    <div>
      <main ref={container} className="relative h-[200vh]">
        <HomeSection scrollYProgress={scrollYProgress} />
        <MainSection scrollYProgress={scrollYProgress} />
      </main>
      <DocumentationPage />
      <Footer />
    </div>
  );
}

// Enhanced HomeSection component
const HomeSection: React.FC<SectionProps> = ({ scrollYProgress }) => {
  const scale = useTransform(scrollYProgress, [0, 1], [1, 0.8]);
  const rotate = useTransform(scrollYProgress, [0, 1], [0, -5]);

  return (
    <motion.div
      style={{ scale, rotate }}
      className="sticky top-0 h-screen bg-[#121212] text-[#e0cbbd] font-serif px-6 py-12 flex flex-col items-center justify-center"
    >
      <div className="max-w-5xl mx-auto text-center">
        <h1 className="text-6xl md:text-7xl tracking-wide mb-4 font-serif">
          Welcome to
        </h1>
        <h2 className="text-5xl md:text-6xl font-bold text-white mb-6 flex items-center justify-center">
          <span className="text-[#e2725b] text-4xl italic mr-2">தமிழ்</span> OCR
        </h2>

        <p className="text-center text-[#ccc] max-w-2xl mx-auto text-lg mb-10">
          Extract and digitize Tamil text from images instantly with our
          advanced optical character recognition technology.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12 max-w-4xl mx-auto">
          <div className="bg-[#1d1d1d] p-6 rounded-xl border border-[#333] hover:border-[#e2725b] transition">
            <div className="text-[#e2725b] text-4xl mb-3">01</div>
            <h3 className="text-white text-xl font-medium mb-2">
              Upload Image
            </h3>
            <p className="text-[#aaa]">
              Select and upload any image containing Tamil text that you need to
              extract.
            </p>
          </div>

          <div className="bg-[#1d1d1d] p-6 rounded-xl border border-[#333] hover:border-[#e2725b] transition">
            <div className="text-[#e2725b] text-4xl mb-3">02</div>
            <h3 className="text-white text-xl font-medium mb-2">Process</h3>
            <p className="text-[#aaa]">
              Our advanced AI algorithms analyze and extract text with high
              accuracy.
            </p>
          </div>

          <div className="bg-[#1d1d1d] p-6 rounded-xl border border-[#333] hover:border-[#e2725b] transition">
            <div className="text-[#e2725b] text-4xl mb-3">03</div>
            <h3 className="text-white text-xl font-medium mb-2">Get Results</h3>
            <p className="text-[#aaa]">
              Copy your extracted text for immediate use in documents,
              translation, or analysis.
            </p>
          </div>
        </div>

        <div className="bg-[#1a1a1a] p-8 rounded-xl border border-[#333] max-w-3xl mx-auto">
          <h3 className="text-white text-2xl mb-4">
            Why Choose Our Tamil OCR?
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
            <div className="flex items-start">
              <div className="text-[#e2725b] mr-3 text-xl">✓</div>
              <div>
                <h4 className="text-white font-medium">High Accuracy</h4>
                <p className="text-[#aaa] text-sm">
                  Advanced processing for complex Tamil characters
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="text-[#e2725b] mr-3 text-xl">✓</div>
              <div>
                <h4 className="text-white font-medium">Privacy First</h4>
                <p className="text-[#aaa] text-sm">
                  Your uploads are processed securely
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="text-[#e2725b] mr-3 text-xl">✓</div>
              <div>
                <h4 className="text-white font-medium">Fast Processing</h4>
                <p className="text-[#aaa] text-sm">Get your text in seconds</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="text-[#e2725b] mr-3 text-xl">✓</div>
              <div>
                <h4 className="text-white font-medium">Clean Interface</h4>
                <p className="text-[#aaa] text-sm">
                  Simple, distraction-free experience
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-10">
          <button
            onClick={() =>
              window.scrollTo({ top: window.innerHeight, behavior: "smooth" })
            }
            className="bg-[#e2725b] hover:bg-[#d56451] text-white px-8 py-3 rounded-full font-medium transition"
          >
            Get Started Now
          </button>
        </div>
      </div>
    </motion.div>
  );
};

// Also enhance the MainSection for better consistency
const MainSection: React.FC<SectionProps> = ({ scrollYProgress }) => {
  const scale = useTransform(scrollYProgress, [0, 1], [0.8, 1]);
  const rotate = useTransform(scrollYProgress, [0, 1], [5, 0]);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ResponseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      setSelectedFile(file);

      // Create preview URL
      const fileUrl = URL.createObjectURL(file);
      setPreviewUrl(fileUrl);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select an input file/image");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      // Using empty string as in your original code
      const res = await fetch("", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to upload file");
      }

      const data = await res.json();
      setResponse(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
      setResponse(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Clean up preview URL when component unmounts
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <motion.div
      style={{ scale, rotate }}
      className="relative h-screen bg-[#1b1b1b] text-[#f2f2f2] font-sans"
    >
      <main className="flex flex-col items-center justify-center py-24 px-6">
        <h1 className="text-5xl font-serif text-center tracking-wide">
          Start Your <span className="text-[#e2725b] italic">Recognition</span>{" "}
          Journey
        </h1>
        <p className="mt-4 text-lg text-gray-400 max-w-xl text-center">
          Extract Tamil text from images with ease. Clean interface. Private. No
          distractions.
        </p>

        <div className="mt-12 bg-[#2c2c2c] p-8 rounded-2xl shadow-lg w-full max-w-xl">
          <form onSubmit={handleSubmit}>
            <div className="mb-6">
              <label className="block text-sm mb-2 font-semibold">
                Upload Image
              </label>
              <div className="border-2 border-dashed border-[#444] rounded-lg p-6 text-center hover:border-[#e2725b] transition">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center"
                >
                  {previewUrl ? (
                    <Image
                      src={previewUrl}
                      alt="Preview"
                      className="max-h-48 max-w-full mb-4 rounded"
                    />
                  ) : (
                    <div className="text-4xl text-[#555] mb-3">📷</div>
                  )}
                  <span className="bg-[#3a3a3a] text-white py-2 px-4 rounded-md hover:bg-[#444] transition">
                    {selectedFile ? "Change Image" : "Select Image"}
                  </span>
                  {selectedFile && (
                    <p className="mt-2 text-sm text-gray-400">
                      {selectedFile.name}
                    </p>
                  )}
                </label>
              </div>
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className={`w-full ${
                isLoading ? "bg-gray-600" : "bg-rose-600 hover:bg-rose-700"
              } transition text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center`}
            >
              {isLoading ? (
                <>
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Processing...
                </>
              ) : (
                "Extract Text"
              )}
            </button>
          </form>

          {response && response.text && (
            <div className="mt-8">
              <h3 className="font-medium text-lg mb-2">Extracted Text:</h3>
              <div className="bg-[#1f1f1f] text-white p-4 rounded-lg min-h-[100px] break-words">
                <div className="mb-4 font-mono overflow-x-auto">
                  {JSON.stringify(response, null, 2)}
                </div>

                <div className="border-t border-[#333] pt-4 mt-4">
                  <h4 className="text-sm uppercase text-gray-500 mb-2">
                    Extracted Tamil Words:
                  </h4>
                  <div className="flex flex-wrap gap-2 text-lg font-sans">
                    {Array.isArray(response.text) ? (
                      response.text.map((word, index) => (
                        <span
                          key={index}
                          className="bg-[#2a2a2a] px-3 py-1 rounded-md hover:bg-[#333] transition"
                        >
                          {word}
                        </span>
                      ))
                    ) : (
                      <p>No text detected in image.</p>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex justify-end mt-2">
                <button
                  onClick={() => {
                    const textContent = Array.isArray(response.text)
                      ? response.text.join(" ")
                      : "";
                    navigator.clipboard.writeText(textContent);
                  }}
                  className="text-sm bg-[#333] hover:bg-[#444] text-white px-3 py-1 rounded transition flex items-center"
                >
                  <span className="mr-1">Copy All</span>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <rect
                      x="9"
                      y="9"
                      width="13"
                      height="13"
                      rx="2"
                      ry="2"
                    ></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-red-900/20 border border-red-900/40 text-red-400 p-3 rounded">
              {error}
            </div>
          )}
        </div>
      </main>
    </motion.div>
  );
};
