'use client';
import {motion, useScroll, useTransform} from 'framer-motion';
import Lenis from 'lenis';
import React, { useEffect,  useRef, useState} from 'react' 


export default function Home() {

    const container = useRef(null);

    const { scrollYProgress } = useScroll({
        target: container,
        offset: ["start start", "end end"]
    })

    useEffect(() => {
        const lenis = new Lenis();

        function raf(time: number) {
            lenis.raf(time)
            requestAnimationFrame(raf)
        }

        requestAnimationFrame(raf)
    }, [])


    return(
      <div>
    <main ref={container} className="relative h-[200vh]">
        <HomeSection scrollYProgress={scrollYProgress} />
        <MainSection scrollYProgress={scrollYProgress} />
        </main>
        <Footer />
        </div>
    );
}

const HomeSection = ({scrollYProgress}: any) => {
    const scale = useTransform(scrollYProgress, [0, 1], [1, 0.8]);
    const rotate = useTransform(scrollYProgress, [0,1], [0, -5]);

    return(
      <motion.div style={{scale, rotate}} className="sticky top-0 h-screen bg-[#121212] text-[#e0cbbd] font-serif px-6 py-16 flex flex-col items-center justify-center">
      <h1 className="text-5xl md:text-6xl tracking-wide mb-4">welcome to</h1>
      <h2 className="text-4xl md:text-5xl font-bold text-white mb-4"><span className="text-[#e2725b] text-3xl italic">தமிழ்</span> OCR</h2>
      <p className="text-center text-[#ccc] max-w-md mb-10">
        Recognize Tamil text from images instantly. Upload your file, then click Get Text to extract the text.
      </p>
    </motion.div>
    );
}

const MainSection = ({scrollYProgress}: any) => {
    const scale = useTransform(scrollYProgress, [0, 1], [0.8, 1]);
    const rotate = useTransform(scrollYProgress, [0, 1], [5, 0])

    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [response, setResponse] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
 
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if(event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if(!selectedFile) {
      setError("Pelese select an input file/image");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('', {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to upload an file");
      }

      const data = await res.json();
      setResponse(data);
      setError(null);
    } catch(err: any) {
        setError(err.message || "An unexpected erroe Occured");
        setResponse(null);
    }

  };

    return(
        <motion.div style={{scale, rotate}} className="relative h-screen bg-[#1b1b1b] text=[#f2f2f2] font-sans">
        <main className="flex flex-col items-center justify-center py-24 px-6">
        <h1 className="text-5xl font-serif text-center tracking-wide">Start Your <span className="text-[#e2725b] italic">Recogination</span> Journey</h1>
        <p className="mt-4 text-lg text-gray-400 max-w-xl text-center">
          Extract Tamil text from images with ease. Clean interface. Private. No distractions.
        </p>
 
        <div className="mt-12 bg-[#2c2c2c] p-8 rounded-2xl shadow-lg w-full max-w-xl">
          <form onSubmit={handleSubmit}>
          <label className="block text-sm mb-2 font-semibold">Upload Image</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-white bg-[#3a3a3a] border border-[#555] rounded-md cursor-pointer px-4 py-2 mb-6"
          />
          <button type='submit' className='w-full bg-rose-600 hover:bg-rose-700 transition text-white font-semibold py-2  px-4 rounded-lg'>Submit</button>
 
          <label className="block text-gray-400 my-2 font-semibold">Output</label>
          </form>
          <div className="bg-[#1f1f1f] text-white p-4 rounded-lg font-mono min-h-[100px]">
            {JSON.stringify(response, null, 2) || 'No output yet.'}
          </div>
        </div>
      </main>
        </motion.div>
    )
}

export  function Footer() {
  return (
    <div  
    className='relative h-[800px]'
    style={{clipPath: "polygon(0% 0, 100% 0%, 100% 100%, 0 100%)"}}
  >
    <div className='fixed bottom-0 h-[800px] w-full'>
      <Content />
    </div>
  </div>
  )
}

export  function Content() {
  return (
    <div className='bg-[#4E4E5A] py-8 px-12 h-full w-full flex flex-col justify-between'>
        <Section1 />
        <Section2 />
    </div>
  )
}

const Section1 = () => {
    return (
        <div>
            <Nav />
        </div>
    )
}

const Section2 = () => {
    return (
        <div className='flex justify-between items-end'>
            <h1 className='text-[14vw] leading-[0.8] mt-10'>Tamil Ocr</h1>
            <p>©copyright</p>
        </div>
    )
}

const Nav = () => {
    return (
        <div className='flex shrink-0 gap-20'>
            <div className='flex flex-col gap-2'>
                <h3 className='mb-2 uppercase text-[#ffffff80]'>About</h3>
                <p>Home</p>
                <p>Projects</p>
                <p>Our Mission</p>
                <p>Contact Us</p>
            </div>
            <div className='flex flex-col gap-2'>
                <h3 className='mb-2 uppercase text-[#ffffff80]'>Education</h3>
                <p>News</p>
                <p>Learn</p>
                <p>Certification</p>
                <p>Publications</p>
            </div>
        </div>
    )
}

