'use client'
import React from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  const handleDiagnosis = () => {
    router.push('/diagnosis'); 
  };

  return (
    <div className="bg-white text-black flex items-center justify-center h-screen">
      <div className="flex flex-col items-center justify-center">
        <h1 className="font-bold lg:text-6xl md:text-4xl sm:text-3xl">
          Oral Cancer Detection Toolkit
        </h1>
        <button
          className="rounded bg-black text-white font-bold border-2 p-4 m-16 hover:bg-white hover:text-black hover:border-black"
          onClick={handleDiagnosis}
        >
          Get Diagnosis
        </button>
      </div>
    </div>
  );
}
