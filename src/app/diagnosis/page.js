'use client';

import React, { useState } from 'react';
import axios from 'axios';

export default function Diagnosis() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageUpload = (event) => {
    setImage(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("An error occurred while processing the image.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-black text-white p-4">
      <h1 className="text-4xl font-extrabold mb-6 text-center">Upload an Image for Diagnosis</h1>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="mb-6 p-2 bg-gray-800 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-white"
      />
      <button
        onClick={handleSubmit}
        className="rounded bg-black text-white font-bold border-2 p-4 m-16 hover:bg-white hover:text-black hover:border-black"
      >
        Get Diagnosis
      </button>
      {result && (
        <div className="mt-6 text-center bg-gray-800 p-6 rounded-md shadow-md">
          <h2 className="text-2xl font-semibold mb-4">Diagnosis Result</h2>
          <p className="text-lg mb-2">Diagnosis: {result.diagnosis}</p>
          <p className="text-lg">Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}
