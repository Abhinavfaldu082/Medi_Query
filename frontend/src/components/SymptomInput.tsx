// src/components/SymptomInput.tsx
import React, { useState, type FormEvent } from "react";
import AudioInputButton from './AudioInputButton';

interface SymptomInputProps {
  onSubmit: (symptoms: string) => void;
  isLoading: boolean;
  value: string; 
  onChange: (value: string) => void; 
}

const SymptomInput: React.FC<SymptomInputProps> = ({
  onSubmit,
  isLoading,
  value,
  onChange,
}) => {
  const [symptoms, setSymptoms] = useState<string>("");
  const [audioError, setAudioError] = useState<string | null>(null);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!symptoms.trim() || isLoading) return;
    onSubmit(symptoms);
  };
  const handleTranscription = (text: string) => {
    setSymptoms((prev) => (prev ? `${prev} ${text}` : text)); // Append or set
    setAudioError(null);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      {" "}
      {/* Reduced space-y */}
      <div>
        <label
          htmlFor="symptomDescription"
          className="block text-sm font-medium text-slate-300 mb-1"
        >
          Describe your symptoms:
        </label>
        <textarea
          id="symptomDescription"
          name="symptoms"
          rows={5}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Type symptoms or use microphone"
          disabled={isLoading}
          className="w-full p-3 bg-slate-700 rounded-md border border-slate-600 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none placeholder-slate-500 disabled:opacity-60"
          required
        />
      </div>
      <AudioInputButton
        onTranscriptionComplete={handleTranscription}
        onTranscriptionError={setAudioError}
        apiBaseUrl={import.meta.env.VITE_API_URL || "http://localhost:8000"}
        disabled={isLoading}
      />
      {audioError && (
        <p className="text-xs text-red-400 -mt-2 mb-2">{audioError}</p>
      )}
      <button
        type="submit"
        disabled={isLoading || !symptoms.trim()}
        className="w-full bg-sky-600 hover:bg-sky-700 text-white font-semibold py-3 px-4 rounded-md transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? "Analyzing..." : "Explore Symptoms"}
      </button>
    </form>
  );
};

export default SymptomInput;