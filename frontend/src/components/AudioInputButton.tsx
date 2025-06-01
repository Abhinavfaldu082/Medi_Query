// src/components/AudioInputButton.tsx
import React, { useState, useRef } from "react";

interface AudioInputButtonProps {
  onTranscriptionComplete: (text: string) => void;
  onTranscriptionError: (error: string) => void;
  apiBaseUrl: string;
  disabled?: boolean;
}

const AudioInputButton: React.FC<AudioInputButtonProps> = ({
  onTranscriptionComplete,
  onTranscriptionError,
  apiBaseUrl,
  disabled,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    setAudioError(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const msg = "getUserMedia not supported on your browser!";
      setAudioError(msg);
      onTranscriptionError(msg);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream); // Default MIME type is often webm/opus or ogg/opus

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        // Determine MIME type - browser default, or specify if possible
        // For broader compatibility, stick with what the browser provides by default.
        // Whisper via FFmpeg should handle common types like webm/opus.
        const mimeType = mediaRecorderRef.current?.mimeType || "audio/webm";
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        audioChunksRef.current = []; // Clear chunks for next recording

        if (!audioBlob || audioBlob.size === 0) {
          throw new Error("No audio recorded or audio blob is empty.");
        }

        console.log(
          "Audio Blob size:",
          audioBlob.size,
          "type:",
          audioBlob.type
        );

        const formData = new FormData();
        // Give it a generic extension, FFmpeg will figure it out.
        // Or use an extension based on mimeType if known (e.g., .webm)
        formData.append(
          "audio_file",
          audioBlob,
          `myaudio.${mimeType.split("/")[1] || "webm"}`
        );

        setIsTranscribing(true);
        try {
          const response = await fetch(`${apiBaseUrl}/audio_input`, {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(
              errorData.detail || `Transcription failed: ${response.status}`
            );
          }

          const result = await response.json();
          if (result.transcription) {
            onTranscriptionComplete(result.transcription);
          } else {
            throw new Error("Transcription result did not contain text.");
          }
        } catch (e: any) {
          console.error("Error transcribing:", e);
          setAudioError(e.message || "Failed to process audio.");
          onTranscriptionError(e.message || "Failed to process audio.");
        } finally {
          setIsTranscribing(false);
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (e: any) {
      console.error("Error starting recording:", e);
      const msg =
        "Could not start recording. Check microphone permissions or ensure a microphone is connected.";
      setAudioError(msg);
      onTranscriptionError(msg);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop(); // This will trigger the 'onstop' handler
      setIsRecording(false);
      // Transcription state is handled in onstop
    }
  };

  const handleButtonClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  let buttonText = "Start Recording";
  let buttonColor = "bg-green-600 hover:bg-green-700";
  if (isRecording) {
    buttonText = "Stop Recording";
    buttonColor = "bg-red-600 hover:bg-red-700";
  } else if (isTranscribing) {
    buttonText = "Transcribing...";
    buttonColor = "bg-yellow-600 hover:bg-yellow-700"; // Or keep it disabled looking
  }

  return (
    <div className="my-2">
      <button
        type="button"
        onClick={handleButtonClick}
        disabled={disabled || isTranscribing} // Disable while transcribing
        className={`w-full ${buttonColor} text-white font-semibold py-2.5 px-4 rounded-md transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2`}
      >
        <span>{buttonText}</span>
      </button>
      {audioError && <p className="text-xs text-red-400 mt-1">{audioError}</p>}
    </div>
  );
};

export default AudioInputButton;
