"use client";
import { BarChart3, Brain, LayoutDashboard } from "lucide-react";

export default function Sidebar() {
  return (
    <div className="w-64 h-screen bg-slate-900 p-6 flex flex-col gap-6 border-r border-slate-700">

      <h1 className="text-xl font-bold">AI Inventory</h1>

      <nav className="space-y-4 text-slate-300">
        <div className="flex items-center gap-2 hover:text-white cursor-pointer">
          <LayoutDashboard size={18}/> Dashboard
        </div>
        <div className="flex items-center gap-2 hover:text-white cursor-pointer">
          <BarChart3 size={18}/> Predictions
        </div>
        <div className="flex items-center gap-2 hover:text-white cursor-pointer">
          <Brain size={18}/> Insights
        </div>
      </nav>

    </div>
  );
}