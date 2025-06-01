// src/components/ResultsDisplay.tsx
import React from "react";

interface Source {
  title: string;
  url: string;
}

interface QAResult {
  type: "qa";
  question: string;
  userInput?: string; // User's original input if different from question
  answer: string;
  sources: Source[];
  // retrieved_context_preview could also be displayed if desired
}

interface SymptomExplorationItem {
  symptom_category: string;
  related_question_asked: string;
  information: string;
  sources: Source[];
}

interface SymptomExploreResult {
  type: "symptom_explore";
  input_text: string;
  identified_symptoms_for_query: string[];
  exploration_results: SymptomExplorationItem[] | string[]; // string[] for "Could not identify" case
  disclaimer: string;
}

type ResultsData = QAResult | SymptomExploreResult;

interface ResultsDisplayProps {
  results: ResultsData;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results }) => {
  if (!results) return null;

  return (
    <div className="space-y-6">
      {results.type === "qa" && (
        <div className="p-4 bg-slate-700 rounded-md shadow">
          <h3 className="text-lg font-semibold text-sky-300 mb-2">
            Answer to:{" "}
            <span className="italic text-slate-200">
              "{results.userInput || results.question}"
            </span>
          </h3>
          <p className="text-slate-200 whitespace-pre-line mb-3">
            {results.answer}
          </p>
          {results.sources && results.sources.length > 0 && (
            <div>
              <h4 className="text-md font-semibold text-slate-300 mb-1">
                Sources:
              </h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                {results.sources.map((source, index) => (
                  <li key={index} className="text-slate-400">
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-400 hover:text-sky-300 hover:underline"
                    >
                      {source.title}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {results.type === "symptom_explore" && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-sky-300 mb-1">
              Symptom Exploration for:
            </h3>
            <p className="italic text-slate-300 bg-slate-700 p-3 rounded-md">
              "{results.input_text}"
            </p>
          </div>

          {results.identified_symptoms_for_query &&
            results.identified_symptoms_for_query.length > 0 && (
              <p className="text-sm text-slate-400">
                Based on your description, we explored information related to:{" "}
                {results.identified_symptoms_for_query.join(", ")}.
              </p>
            )}

          {Array.isArray(results.exploration_results) &&
          results.exploration_results.length > 0 ? (
            typeof results.exploration_results[0] === "string" ? (
              <p className="text-slate-300 bg-amber-700/30 p-3 rounded-md">
                {results.exploration_results[0]}
              </p>
            ) : (
              (results.exploration_results as SymptomExplorationItem[]).map(
                (item, index) => (
                  <div
                    key={index}
                    className="p-4 bg-slate-700 rounded-md shadow"
                  >
                    <h4 className="text-md font-semibold text-sky-400 mb-2">
                      Regarding:{" "}
                      <span className="text-slate-200">
                        {item.symptom_category}
                      </span>
                    </h4>
                    {/* Optional: Display the question asked to RAG
                        <p className="text-xs text-slate-500 mb-1 italic">
                            (Queried as: "{item.related_question_asked}")
                        </p> 
                        */}
                    <p className="text-slate-200 whitespace-pre-line mb-3">
                      {item.information}
                    </p>
                    {item.sources && item.sources.length > 0 && (
                      <div>
                        <h5 className="text-sm font-semibold text-slate-400 mb-1">
                          Potential Sources:
                        </h5>
                        <ul className="list-disc list-inside space-y-1 text-xs">
                          {item.sources.map((source, sIndex) => (
                            <li key={sIndex} className="text-slate-500">
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-sky-500 hover:text-sky-400 hover:underline"
                              >
                                {source.title}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )
              )
            )
          ) : null}

          {results.disclaimer && (
            <p className="mt-4 text-xs text-slate-500 italic border-t border-slate-700 pt-3">
              {results.disclaimer}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
