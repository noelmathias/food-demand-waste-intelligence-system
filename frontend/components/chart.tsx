"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

export default function Chart({ demand, waste }: any) {
  const data = [
    { name: "Demand", value: demand || 0 },
    { name: "Waste", value: waste || 0 },
  ];

  return (
    <div className="bg-slate-900 p-5 rounded-xl">
      <h3 className="mb-3 font-semibold">Insights Chart</h3>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid stroke="#334155" />
          <XAxis dataKey="name" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Bar dataKey="value" fill="#8b5cf6" radius={[6,6,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}