"use client";
import { motion } from "framer-motion";

export default function StatCard({ title, value, color }: any) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-5 rounded-xl ${color} shadow-lg`}
    >
      <p className="text-sm">{title}</p>
      <h2 className="text-2xl font-bold">{value}</h2>
    </motion.div>
  );
}