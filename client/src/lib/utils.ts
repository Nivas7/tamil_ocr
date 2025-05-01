import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { useState, useEffect } from "react";


export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}


const useMousePosition= ()=> {
  const [mousePosition, setMousePosition] = useState({ x: null, y: null });
 
  const updateMousePosition = e => {
    setMousePosition({ x: e.clientX, y: e.clientY });
  };
 
  useEffect(() => {
    window.addEventListener("mousemove", updateMousePosition);
 
    return () => window.removeEventListener("mousemove", updateMousePosition);
  }, []);
 
  return mousePosition;
};
 
export default useMousePosition;
