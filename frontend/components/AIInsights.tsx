"use client";
import { motion } from "framer-motion";

export default function AIInsights({ demand, waste }: any) {

  if (!demand || !waste) return null;

  const diff = ((waste - demand) / demand) * 100;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-slate-900 p-5 rounded-xl space-y-2"
    >
      <h3 className="font-semibold text-lg">AI Explanation</h3>

      <p>
        Demand is <span className="text-blue-400">{demand}</span> units while waste is{" "}
        <span className="text-red-400">{waste}</span>.
      </p>

      {diff > 0 ? (
        <p className="text-yellow-400">
          ⚠️ Waste exceeds demand by {diff.toFixed(1)}%. Reduce production.
        </p>
      ) : (
        <p className="text-green-400">
          ✅ Demand is higher than waste. Production is efficient.
        </p>
      )}

      <p className="text-slate-400 text-sm">
        AI suggests adjusting production dynamically to minimize loss and optimize inventory.
      </p>
    </motion.div>
  );
}