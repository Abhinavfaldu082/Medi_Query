// src/components/QueryInput.tsx
import React, { useState } from "react"; // Keep useState for audioError
import type { FormEvent } from "react";
import AudioInputButton from "./AudioInputButton";

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  value: string; // This is the text for the input field, controlled by App.tsx
  onChange: (value: string) => void; // This updates the text in App.tsx
}

const QueryInput: React.FC<QueryInputProps> = ({
  onSubmit,
  isLoading,
  value,
  onChange,
}) => {
  const [audioError, setAudioError] = useState<string | null>(null);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!value.trim() || isLoading) return; // Use prop 'value'
    onSubmit(value);
  };

  // This function is called when AudioInputButton completes transcription
  const handleAudioTranscription = (text: string) => {
    onChange(value ? `${value}\n${text}`.trim() : text); // Update text in App.tsx
    setAudioError(null);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div>
        <label
          htmlFor="healthQuery"
          className="block text-sm font-medium text-slate-300 mb-1"
        >
          Your Question:
        </label>
        <input
          id="healthQuery"
          name="query"
          type="text"
          value={value} // Controlled by App.tsx
          onChange={(e) => onChange(e.target.value)} // Updates App.tsx state
          placeholder="Type your question or use microphone"
          disabled={isLoading}
          className="w-full p-3 bg-slate-700 rounded-md border border-slate-600 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none placeholder-slate-500 disabled:opacity-60"
          required
        />
      </div>

      <AudioInputButton
        onTranscriptionComplete={handleAudioTranscription}
        onTranscriptionError={setAudioError}
        apiBaseUrl={import.meta.env.VITE_API_URL || "http://localhost:8000"}
        disabled={isLoading}
      />
      {audioError && (
        <p className="text-xs text-red-400 -mt-2 mb-2">{audioError}</p>
      )}

      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        className="w-full bg-sky-600 hover:bg-sky-700 text-white font-semibold py-3 px-4 rounded-md transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? "Searching..." : "Search Health Q&A"}
      </button>
    </form>
  );
};

export default QueryInput;