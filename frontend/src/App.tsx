// src/App.tsx
import React, { useState, useEffect } from "react";
import QueryInput from "./components/QueryInput";
import SymptomInput from "./components/SymptomInput";
import ResultsDisplay from "./components/ResultsDisplay";
import ReportButton from "./components/ReportButton";
import ImageInputArea from "./components/ImageInputArea";

type View = "qa" | "symptom";

interface ResultItem {
  type: "qa" | "symptom_explore";
  data: any; // The actual response data from backend
  queryOrSymptoms: string; // The input that generated this result
}

function App() {
  const [currentView, setCurrentView] = useState<View>("qa");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [qaResults, setQaResults] = useState<ResultItem | null>(null);
  const [symptomResults, setSymptomResults] = useState<ResultItem | null>(null);

  const [error, setError] = useState<string | null>(null);

  // State for controlled input components
  const [currentQueryText, setCurrentQueryText] = useState("");
  const [currentSymptomsText, setCurrentSymptomsText] = useState("");

  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const handleClearCurrentViewResults = () => {
    if (currentView === "qa") {
      setQaResults(null);
      setCurrentQueryText(""); // Clear input text for QA
    } else if (currentView === "symptom") {
      setSymptomResults(null);
      setCurrentSymptomsText(""); // Clear input text for Symptoms
    }
    setError(null);
  };

  const handleQuerySubmit = async (query: string) => {
    // query argument now comes from QueryInput's internal state via its onSubmit
    if (!query.trim()) {
      // Use the passed query
      setError("Please enter a question.");
      return; // Do not clear qaResults here, let Clear button do it or new search replace it
    }
    setIsLoading(true);
    setError(null);
    setQaResults(null); // Clear previous results for a new search

    const formData = new FormData();
    formData.append("question", query);

    try {
      const response = await fetch(`${API_BASE_URL}/qa`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      setQaResults({ type: "qa", data: data, queryOrSymptoms: query });
    } catch (e: any) {
      console.error("Error fetching QA:", e);
      setError(e.message || "Failed to fetch Q&A results.");
      setQaResults({ type: "qa", data: null, queryOrSymptoms: query });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSymptomSubmit = async (symptoms: string) => {
    // symptoms argument now comes from SymptomInput's internal state via its onSubmit
    if (!symptoms.trim()) {
      // Use the passed symptoms
      setError("Please describe your symptoms.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setSymptomResults(null);

    const formData = new FormData();
    formData.append("symptoms_text", symptoms);

    try {
      const response = await fetch(`${API_BASE_URL}/symptom_explore`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }
      const data = await response.json();
      setSymptomResults({
        type: "symptom_explore",
        data: data,
        queryOrSymptoms: symptoms,
      });
    } catch (e: any) {
      console.error("Error fetching symptom exploration:", e);
      setError(e.message || "Failed to fetch symptom exploration results.");
      setSymptomResults({
        type: "symptom_explore",
        data: null,
        queryOrSymptoms: symptoms,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    const resultsToReport = currentView === "qa" ? qaResults : symptomResults;
    if (!resultsToReport || !resultsToReport.data) {
      setError("No data available in the current view to generate a report.");
      return;
    }
    try {
      const response = await fetch(`${API_BASE_URL}/generate_report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(resultsToReport.data),
      });
      if (!response.ok) {
        let errorDetail = `Report generation failed: ${response.status}`;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (parseError) {
          errorDetail = response.statusText || errorDetail;
        }
        throw new Error(errorDetail);
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "MediQuery_AI_Report.pdf";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (e: any) {
      console.error("Error generating report:", e);
      setError(e.message || "Failed to generate report.");
    }
  };

  useEffect(() => {
    setError(null); // Clear global error when view changes
  }, [currentView]);

  const handleOcrTextReceived = (text: string) => {
    console.log("App.tsx: handleOcrTextReceived with text:", `"${text}"`);
    if (currentView === "qa") {
      setCurrentQueryText(text);
    } else if (currentView === "symptom") {
      setCurrentSymptomsText(text);
    }
  };

  const currentResultsToDisplay =
    currentView === "qa" ? qaResults : symptomResults;
  const displayError = error;

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center p-4 md:p-8 font-sans">
      <header className="w-full max-w-3xl mb-8 text-center">
        <h1 className="text-4xl md:text-5xl font-bold text-sky-400">
          MediQuery AI
        </h1>
        <p className="text-slate-400 mt-2">
          Your AI-powered health information assistant
        </p>
      </header>

      <div className="w-full max-w-3xl bg-slate-800 shadow-2xl rounded-lg p-6 md:p-8">
        <div className="flex border-b border-slate-700 mb-6">
          <div>
            <button
              onClick={() => {
                setCurrentView(
                  "qa"
                ); /* setError(null); No longer needed here due to useEffect */
              }}
              className={`py-3 px-6 font-semibold transition-colors duration-150 ${
                currentView === "qa"
                  ? "border-b-2 border-sky-500 text-sky-400"
                  : "text-slate-400 hover:text-sky-300"
              }`}
            >
              Health Q&A
            </button>
            <button
              onClick={() => {
                setCurrentView(
                  "symptom"
                ); /* setError(null); No longer needed here due to useEffect */
              }}
              className={`py-3 px-6 font-semibold transition-colors duration-150 ${
                currentView === "symptom"
                  ? "border-b-2 border-sky-500 text-sky-400"
                  : "text-slate-400 hover:text-sky-300"
              }`}
            >
              Symptom Explorer
            </button>
          </div>
        </div>

        {currentView === "qa" && (
          <div className="animate-fadeIn space-y-4">
            <h2 className="text-2xl font-semibold text-slate-200">
              Ask a Health Question
            </h2>
            <QueryInput
              onSubmit={handleQuerySubmit}
              isLoading={isLoading && currentView === "qa"} // isLoading specific to this view's action
              value={currentQueryText}
              onChange={setCurrentQueryText}
            />
            <ImageInputArea
              onOcrComplete={handleOcrTextReceived}
              onOcrError={(err) => setError(`OCR Error: ${err}`)}
              apiBaseUrl={API_BASE_URL}
              disabled={isLoading}
            />
            {/* Show Clear button only if there are results or an error for THIS view */}
            {(qaResults ||
              (error &&
                currentView === "qa" &&
                !isLoading &&
                !qaResults?.data)) && (
              <button
                onClick={handleClearCurrentViewResults}
                className="w-full mt-2 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2.5 px-4 rounded-md transition-colors duration-150"
              >
                Clear Q&A Results
              </button>
            )}
          </div>
        )}

        {currentView === "symptom" && (
          <div className="animate-fadeIn space-y-4">
            <h2 className="text-2xl font-semibold text-slate-200">
              Describe Your Symptoms
            </h2>
            <SymptomInput
              onSubmit={handleSymptomSubmit}
              isLoading={isLoading && currentView === "symptom"} // isLoading specific to this view's action
              value={currentSymptomsText}
              onChange={setCurrentSymptomsText}
            />
            <ImageInputArea
              onOcrComplete={handleOcrTextReceived}
              onOcrError={(err) => setError(`OCR Error: ${err}`)}
              apiBaseUrl={API_BASE_URL}
              disabled={isLoading}
            />
            {/* Show Clear button only if there are results or an error for THIS view */}
            {(symptomResults ||
              (error &&
                currentView === "symptom" &&
                !isLoading &&
                !symptomResults?.data)) && (
              <button
                onClick={handleClearCurrentViewResults}
                className="w-full mt-2 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2.5 px-4 rounded-md transition-colors duration-150"
              >
                Clear Symptom Exploration
              </button>
            )}
          </div>
        )}

        <div className="mt-8">
          {isLoading && (
            <div className="flex justify-center items-center p-4">
              <p className="text-sky-400 animate-pulse">Loading...</p>
            </div>
          )}
          {/* Display error if it exists AND we are not currently loading AND (it's a general error OR the current view's results are null) */}
          {displayError &&
            !isLoading &&
            ((currentView === "qa" && (!qaResults || !qaResults.data)) ||
              (currentView === "symptom" &&
                (!symptomResults || !symptomResults.data))) && (
              <div className="mt-4 p-4 bg-red-800/60 border border-red-700 text-red-200 rounded-md">
                <p className="font-semibold text-red-100">An error occurred:</p>
                <p className="text-sm">{displayError}</p>
              </div>
            )}
          {currentResultsToDisplay &&
            currentResultsToDisplay.data &&
            !isLoading && (
              <div className="p-0 animate-fadeIn">
                <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-3">
                  <h3 className="text-xl font-semibold text-sky-400">
                    Information Found
                    {currentResultsToDisplay.queryOrSymptoms && (
                      <span className="text-base font-normal text-slate-400">
                        {" "}
                        for: "{currentResultsToDisplay.queryOrSymptoms}"
                      </span>
                    )}
                  </h3>
                  <ReportButton
                    onClick={handleGenerateReport}
                    isLoading={false} // Assuming PDF generation itself doesn't set global isLoading
                    disabled={!currentResultsToDisplay.data}
                  />
                </div>
                <ResultsDisplay results={currentResultsToDisplay.data} />
              </div>
            )}
        </div>
      </div>

      <footer className="w-full max-w-3xl mt-12 text-center text-sm text-slate-500">
        <p>
          © {new Date().getFullYear()} MediQuery AI. For informational purposes
          only.
        </p>
      </footer>
    </div>
  );
}

export default App;
