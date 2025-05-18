import React, { useState } from "react";
import { motion } from "framer-motion";
import Image from "next/image";

const DocumentationPage = () => {
  const [activeTab, setActiveTab] = useState("about");

  // Team members data - Replace with your actual team information
  const teamMembers = [
    {
      name: "Srinivas R",
      role: "Frontend Developer && Backend With Machine Learning",
      github: "https://github.com/nivas7",
      avatar: null, // Can add avatar image path later
      description:
        "Responsible for React components and UI animations, Backend Server with ML model",
    },
    {
      name: "Rishikesavan A",
      role: "Machine Learning Model Dev && Documentation",
      github: "https://github.com/rishikesavan1",
      avatar: null,
      description: "Implemented the OCR ML model and Tamil text processing",
    },
    {
      name: "Thirumalai S",
      role: "Frontend Developer && UI/UX",
      github: "https://github.com/gunal77",
      avatar: null,
      description: "Created classic elegant UI Components",
    },
    {
      name: "Abu Baker Sithik Rahuman M",
      role: "UI/UX Designer && Poster & Documentation Designer",
      github: "https://github.com/abu28012002",
      avatar: null,
      description: "Created the user interface design and user experience flow",
    },
  ];

  return (
    <div className="min-h-screen bg-[#1b1b1b] text-[#f2f2f2] font-sans py-16 px-6 mt-[15vh]">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto"
      >
        <h1 className="text-4xl font-serif text-center tracking-wide mb-8">
          <span className="text-[#e2725b]">தமிழ்</span> OCR Documentation
        </h1>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-[#2c2c2c] rounded-full p-1.5 inline-flex">
            <button
              onClick={() => setActiveTab("about")}
              className={`px-4 py-2 rounded-full transition ${
                activeTab === "about"
                  ? "bg-[#e2725b] text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              About
            </button>
            <button
              onClick={() => setActiveTab("usage")}
              className={`px-4 py-2 rounded-full transition ${
                activeTab === "usage"
                  ? "bg-[#e2725b] text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              Usage Guide
            </button>
            <button
              onClick={() => setActiveTab("tech")}
              className={`px-4 py-2 rounded-full transition ${
                activeTab === "tech"
                  ? "bg-[#e2725b] text-white"
                  : "text-gray-300 hover:text-white"
              }`}
            >
              Technology
            </button>
          </div>
        </div>

        {/* About Tab */}
        {activeTab === "about" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="bg-[#2c2c2c] rounded-lg p-6 mb-8">
              <h2 className="text-2xl font-serif mb-4 text-[#e0cbbd]">
                Project Overview
              </h2>
              <p className="mb-4">
                The Tamil OCR project is a state-of-the-art optical character
                recognition system specifically designed for the Tamil language.
                Created as a final year project, this application aims to
                provide an accessible, efficient, and accurate solution for
                converting Tamil text in images to editable digital text.
              </p>
              <p>
                Our mission is to preserve and promote Tamil language
                accessibility in the digital age, making it easier for
                researchers, students, and users to digitize Tamil documents and
                content with high accuracy and minimal effort.
              </p>
            </div>

            <h2 className="text-2xl font-serif mb-6 text-center text-[#e0cbbd]">
              Meet Our Team
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {teamMembers.map((member, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="bg-[#2c2c2c] rounded-lg overflow-hidden border border-[#3a3a3a]"
                >
                  <div className="p-6 flex items-center">
                    <div className="w-16 h-16 rounded-full bg-[#3a3a3a] flex items-center justify-center mr-4">
                      {member.avatar ? (
                        <Image
                          src={member.avatar}
                          alt={member.name}
                          className="rounded-full w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-[#e2725b] text-xl font-bold">
                          {member.name.charAt(0)}
                        </span>
                      )}
                    </div>
                    <div>
                      <h3 className="font-medium text-lg">{member.name}</h3>
                      <p className="text-[#e2725b] text-sm">{member.role}</p>
                    </div>
                  </div>
                  <div className="px-6 pb-6">
                    <p className="text-gray-300 mb-4">{member.description}</p>
                    <a
                      href={member.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center text-[#e0cbbd] hover:text-[#e2725b] transition"
                    >
                      <svg
                        className="w-5 h-5 mr-2"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                      </svg>
                      GitHub Profile
                    </a>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Usage Guide Tab */}
        {activeTab === "usage" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-[#2c2c2c] rounded-lg p-6"
          >
            <h2 className="text-2xl font-serif mb-6 text-[#e0cbbd]">
              How to Use Tamil OCR
            </h2>

            <div className="mb-8">
              <h3 className="text-xl font-medium mb-3 text-[#e2725b]">
                1. Upload an Image
              </h3>
              <p className="mb-3">
                Start by uploading an image containing Tamil text. Our system
                works best with:
              </p>
              <ul className="list-disc list-inside pl-4 mb-4 text-gray-300">
                <li>Clear, well-lit images</li>
                <li>Minimal background noise or patterns</li>
                <li>Standard Tamil fonts</li>
                <li>JPG, PNG, or PDF formats</li>
              </ul>
              <div className="bg-[#252525] p-4 rounded-md">
                <p className="text-sm text-gray-400">
                  <strong className="text-[#e0cbbd]">TIP:</strong> For best
                  results, ensure the text in your image is horizontal and not
                  skewed. Higher resolution images typically yield better
                  results.
                </p>
              </div>
            </div>

            <div className="mb-8">
              <h3 className="text-xl font-medium mb-3 text-[#e2725b]">
                2. Process the Image
              </h3>
              <p className="mb-3">
                After uploading your image, click the `&quot`Submit`&quot`
                button to process it. Our system will:
              </p>
              <ul className="list-disc list-inside pl-4 mb-4 text-gray-300">
                <li>Detect text regions in the image</li>
                <li>Apply preprocessing to enhance text clarity</li>
                <li>Recognize the Tamil characters</li>
                <li>Convert them to editable text</li>
              </ul>
              <div className="bg-[#252525] p-4 rounded-md">
                <p className="text-sm text-gray-400">
                  <strong className="text-[#e0cbbd]">NOTE:</strong> Processing
                  time varies with image size and complexity. Please be patient
                  for larger documents.
                </p>
              </div>
            </div>

            <div className="mb-8">
              <h3 className="text-xl font-medium mb-3 text-[#e2725b]">
                3. Review and Edit Results
              </h3>
              <p className="mb-3">Once processing is complete, you can:</p>
              <ul className="list-disc list-inside pl-4 mb-4 text-gray-300">
                <li>Review the extracted text</li>
                <li>Edit any recognition errors</li>
                <li>Copy the text to clipboard</li>
                <li>Save the results to your history</li>
                <li>Download the text as a file</li>
              </ul>
            </div>
          </motion.div>
        )}

        {/* Technology Tab */}
        {activeTab === "tech" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-[#2c2c2c] rounded-lg p-6"
          >
            <h2 className="text-2xl font-serif mb-6 text-[#e0cbbd]">
              Technology Stack
            </h2>

            <div className="mb-8">
              <h3 className="text-xl font-medium mb-3 text-[#e2725b]">
                Frontend Technologies
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                <div className="bg-[#252525] p-4 rounded-md text-center">
                  <div className="text-3xl mb-2">⚛️</div>
                  <h4 className="font-medium">React.js</h4>
                  <p className="text-sm text-gray-400">UI Framework</p>
                </div>
                <div className="bg-[#252525] p-4 rounded-md text-center">
                  <div className="text-3xl mb-2">🎨</div>
                  <h4 className="font-medium">Tailwind CSS</h4>
                  <p className="text-sm text-gray-400">Styling</p>
                </div>
                <div className="bg-[#252525] p-4 rounded-md text-center">
                  <div className="text-3xl mb-2">🎭</div>
                  <h4 className="font-medium">Framer Motion</h4>
                  <p className="text-sm text-gray-400">Animations</p>
                </div>
                <div className="bg-[#252525] p-4 rounded-md text-center">
                  <div className="text-3xl mb-2">🔄</div>
                  <h4 className="font-medium">Next.js</h4>
                  <p className="text-sm text-gray-400">Framework</p>
                </div>
                <div className="bg-[#252525] p-4 rounded-md text-center">
                  <div className="text-3xl mb-2">📜</div>
                  <h4 className="font-medium">Lenis</h4>
                  <p className="text-sm text-gray-400">Smooth Scrolling</p>
                </div>
              </div>
            </div>

            <div className="mb-8">
              <h3 className="text-xl font-medium mb-3 text-[#e2725b]">
                OCR Technology
              </h3>
              <p className="mb-4">
                Our Tamil OCR system uses advanced deep learning models
                specifically trained for Tamil character recognition:
              </p>
              <ul className="list-disc list-inside pl-4 mb-4 text-gray-300">
                <li>
                  Convolutional Neural Networks (CNN) for feature extraction
                </li>
                <li>Recurrent Neural Networks (RNN) for sequence modeling</li>
                <li>Custom-trained models for Tamil language specifics</li>
                <li>
                  Post-processing algorithms for context-aware corrections
                </li>
              </ul>
              <div className="bg-[#252525] p-4 rounded-md">
                <p className="text-sm text-gray-400">
                  <strong className="text-[#e0cbbd]">TECHNICAL NOTE:</strong>{" "}
                  The OCR system is trained on a dataset of over 100,000 Tamil
                  characters and achieves an accuracy of approximately 95% on
                  clear text.
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-xl font-medium mb-3 text-[#e2725b]">
                Backend & Infrastructure
              </h3>
              <p className="mb-4">
                The backend of our application is built with:
              </p>
              <ul className="list-disc list-inside pl-4 text-gray-300">
                <li>Node.js for the server environment</li>
                <li>Express.js for API development</li>
                <li>TensorFlow/PyTorch for running the OCR models</li>
                <li>MongoDB for storing processing history</li>
              </ul>
            </div>
          </motion.div>
        )}

        {/* GitHub Project Link */}
        <div className="mt-12 text-center">
          <a
            href="https://github.com/nivas7/tamil_ocr"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center bg-[#2c2c2c] hover:bg-[#3a3a3a] text-white py-3 px-6 rounded-lg transition"
          >
            <svg
              className="w-5 h-5 mr-2"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            View Project on GitHub
          </a>
        </div>
      </motion.div>
    </div>
  );
};

export default DocumentationPage;
