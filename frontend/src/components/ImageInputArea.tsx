// src/components/ImageInputArea.tsx
import React, { useState } from "react"; // Keep useState
import type { ChangeEvent } from "react";

interface ImageInputAreaProps {
  onOcrComplete: (text: string) => void; // Callback with extracted text
  onOcrError: (error: string) => void;
  apiBaseUrl: string;
  disabled?: boolean;
}

const ImageInputArea: React.FC<ImageInputAreaProps> = ({
  onOcrComplete,
  onOcrError,
  apiBaseUrl,
  disabled,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // State to hold the OCR text locally before user confirms to use it
  const [editableOcrText, setEditableOcrText] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    setEditableOcrText(null); // Clear previous OCR text if new file is selected
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  };

  const handleUploadAndOcr = async () => {
    if (!selectedFile) {
      setUploadError("Please select an image file first.");
      return;
    }

    setIsProcessing(true);
    setUploadError(null);
    setEditableOcrText(null); 

    const formData = new FormData();
    formData.append("image_file", selectedFile, selectedFile.name);

    try {
      const response = await fetch(`${apiBaseUrl}/image_input`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorDetail = `OCR processing failed: ${response.status}`;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (jsonError) {
          const textError = await response.text();
          errorDetail = textError || errorDetail;
        }
        throw new Error(errorDetail);
      }

      const result = await response.json();
      console.log("OCR Raw Result from backend:", result);

      if (result.ocr_results && Array.isArray(result.ocr_results)) {
        let rawLines = result.ocr_results.map((r: { text: string }) => r.text);
        let processedLines = rawLines.map((line) => {
          line = line.replace(/([a-z])([A-Z])/g, "$1 $2");
          line = line.replace(/([A-Z])([A-Z][a-z])/g, "$1 $2");
          return line.trim();
        });

        const extractedText = processedLines.filter((line) => line).join("\n");
        console.log(
          "Locally Extracted Text for editing:",
          `"${extractedText}"`
        );

        if (!extractedText.trim() && result.ocr_results.length === 0) {
          onOcrError("No text could be extracted from the image.");
        } else if (!extractedText.trim() && result.ocr_results.length > 0) {
          onOcrError("Extracted text was empty after processing.");
        } else {
          setEditableOcrText(extractedText); // Set text for editing
        }
      } else {
        throw new Error("OCR result did not contain expected text data.");
      }
    } catch (e: any) {
      console.error("Error uploading or processing image:", e);
      setUploadError(e.message || "Failed to process image.");
      onOcrError(e.message || "Failed to process image.");
    } finally {
      setIsProcessing(false);
    }
    };

    const handleUseEditedText = () => {
      if (editableOcrText !== null) {
        // Check against null, as empty string is a valid edited text
        onOcrComplete(editableOcrText); // Send the (potentially edited) text
        setEditableOcrText(null);
        setSelectedFile(null);
        setPreviewUrl(null);
      }
    };

  return (
    <div className="my-3 p-4 border border-dashed border-slate-600 rounded-lg space-y-3">
      <div>
        <label
          htmlFor="imageUpload"
          className="block text-sm font-medium text-slate-300 mb-1"
        >
          Upload Image for Text Extraction (OCR):
        </label>
        <input
          type="file"
          id="imageUpload"
          accept="image/png, image/jpeg, image/webp"
          onChange={handleFileChange}
          disabled={isProcessing || editableOcrText !== null}
          className="block w-full text-sm text-slate-400
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-md file:border-0
                     file:text-sm file:font-semibold
                     file:bg-sky-700 file:text-sky-100
                     hover:file:bg-sky-600 disabled:opacity-50"
        />
      </div>

      {previewUrl && selectedFile && editableOcrText == null && (
        <div className="mt-2">
          <p className="text-xs text-slate-400 mb-1">
            Preview ({selectedFile.name}):
          </p>
          <img
            src={previewUrl}
            alt="Selected preview"
            className="max-h-40 rounded-md border border-slate-500 object-contain"
          />
        </div>
      )}

      {selectedFile && editableOcrText == null && (
        <button
          type="button"
          onClick={handleUploadAndOcr}
          disabled={isProcessing || disabled}
          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2.5 px-4 rounded-md transition-colors duration-150 disabled:opacity-50"
        >
          {isProcessing ? "Processing Image..." : "Upload"}
        </button>
      )}

      {uploadError && (
        <p className="text-xs text-red-400 mt-1">{uploadError}</p>
      )}

      {editableOcrText !== null && (
        <div className="mt-2 p-3 bg-slate-700/50 rounded-md border border-slate-600 space-y-2">
          <label
            htmlFor="editableOcrTextArea"
            className="text-sm font-medium text-slate-300"
          >
            Edit Extracted Text (from {selectedFile?.name || "uploaded image"}):
          </label>
          <textarea
            id="editableOcrTextArea"
            value={editableOcrText}
            onChange={(e) => setEditableOcrText(e.target.value)}
            rows={6}
            className="w-full p-2 bg-slate-800 text-slate-200 rounded-md border border-slate-500 focus:ring-1 focus:ring-sky-500 outline-none text-xs"
          />
          <button
            onClick={handleUseEditedText}
            className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-3 rounded-md text-sm transition-colors"
          >
            Use this Text in Input Field
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageInputArea;