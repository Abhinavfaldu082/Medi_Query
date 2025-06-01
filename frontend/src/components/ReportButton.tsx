// src/components/ReportButton.tsx
import React from "react";

interface ReportButtonProps {
  onClick: () => void;
  isLoading: boolean; // To disable while report is generating
  disabled?: boolean; // To disable if no results
}

const ReportButton: React.FC<ReportButtonProps> = ({
  onClick,
  isLoading,
  disabled,
}) => {
  return (
    <button
      onClick={onClick}
      disabled={isLoading || disabled}
      className="bg-teal-600 hover:bg-teal-700 text-white font-semibold py-2 px-4 rounded-md text-sm transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {isLoading ? "Generating PDF..." : "Download Report"}
    </button>
  );
};

export default ReportButton;
