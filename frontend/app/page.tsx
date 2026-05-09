"use client"

import { motion } from "framer-motion"
import { useState } from "react"

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "https://food-demand-waste-intelligence-system.onrender.com"

export default function Home() {
  const [store, setStore] = useState("")
  const [item, setItem] = useState("")
  const [day, setDay] = useState("")
  const [month, setMonth] = useState("")
  const [year, setYear] = useState("")
  const [country, setCountry] = useState("")
  const [category, setCategory] = useState("")

  const [demand, setDemand] = useState<number | null>(null)
  const [waste, setWaste] = useState<number | null>(null)
  const [risk, setRisk] = useState<string | null>(null)
  const [production, setProduction] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = async () => {
    try {
      setLoading(true)

      const res = await fetch(`${API_BASE_URL}/predict-all`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          store,
          item,
          day,
          month,
          year,
          country,
          food_category: category,
        }),
      })

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`)
      }

      const data = await res.json()

      setDemand(data.demand)
      setWaste(data.waste)
      setRisk(data.risk)
      setProduction(data.recommended_production)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const generateExplanation = () => {
    if (typeof demand !== "number" || typeof waste !== "number") return null

    let message = ""

    if (waste > demand) {
      message = "Waste is significantly higher than demand, indicating overproduction."
    } else if (demand > waste) {
      message = "Demand exceeds waste, indicating efficient inventory usage."
    } else {
      message = "Demand and waste are balanced."
    }

    const total = demand + waste

    return {
      summary: message,
      gap: Math.abs(waste - demand).toFixed(2),
      efficiency: total > 0 ? ((demand / total) * 100).toFixed(1) : "0",
    }
  }

  const getRiskColor = () => {
    if (risk === "HIGH") return "bg-red-500"
    if (risk === "MEDIUM") return "bg-yellow-500"
    return "bg-green-500"
  }

  const explanation = generateExplanation()

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mx-auto max-w-6xl p-6 text-white"
    >
      <h1 className="mb-6 text-3xl font-bold">AI Inventory Dashboard</h1>

      <div className="mb-6 grid grid-cols-3 gap-4">
        <motion.div whileHover={{ scale: 1.05 }} className="rounded-xl bg-blue-600 p-4">
          <h2>Demand</h2>
          <p className="text-xl font-bold">{typeof demand === "number" ? demand.toFixed(2) : "--"}</p>
        </motion.div>

        <motion.div whileHover={{ scale: 1.05 }} className="rounded-xl bg-red-500 p-4">
          <h2>Waste</h2>
          <p className="text-xl font-bold">{typeof waste === "number" ? waste.toFixed(2) : "--"}</p>
        </motion.div>

        <motion.div whileHover={{ scale: 1.05 }} className="rounded-xl bg-green-500 p-4">
          <h2>Status</h2>
          <p className="text-xl font-bold">Ready</p>
        </motion.div>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-6">
        <div>
          <h2 className="mb-2">Demand Input</h2>

          <input className="mb-2 w-full rounded bg-gray-800 p-2" placeholder="Store" value={store} onChange={(e) => setStore(e.target.value)} />
          <input className="mb-2 w-full rounded bg-gray-800 p-2" placeholder="Item" value={item} onChange={(e) => setItem(e.target.value)} />
          <input className="mb-2 w-full rounded bg-gray-800 p-2" placeholder="Day" value={day} onChange={(e) => setDay(e.target.value)} />
          <input className="mb-2 w-full rounded bg-gray-800 p-2" placeholder="Month" value={month} onChange={(e) => setMonth(e.target.value)} />
          <input className="w-full rounded bg-gray-800 p-2" placeholder="Year" value={year} onChange={(e) => setYear(e.target.value)} />
        </div>

        <div>
          <h2 className="mb-2">Waste Context</h2>

          <input className="mb-2 w-full rounded bg-gray-800 p-2" placeholder="Country" value={country} onChange={(e) => setCountry(e.target.value)} />
          <input className="w-full rounded bg-gray-800 p-2" placeholder="Food Category" value={category} onChange={(e) => setCategory(e.target.value)} />
        </div>
      </div>

      <button
        onClick={handlePredict}
        disabled={loading}
        className="mb-6 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-purple-500 to-indigo-500 p-3"
      >
        {loading && (
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
        )}
        {loading ? "Predicting..." : "Run AI Prediction"}
      </button>

      {typeof demand === "number" && typeof waste === "number" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className={`rounded-xl p-3 font-bold text-white ${getRiskColor()}`}
        >
          Risk Level: {risk}
        </motion.div>
      )}

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-xl bg-gray-800 p-4">
        <h2 className="mb-2 text-lg font-semibold">AI Explanation</h2>

        {explanation ? (
          <>
            <p>{explanation.summary}</p>
            <p className="mt-2 text-sm text-gray-400">Demand-Waste Gap: {explanation.gap}</p>
            <p className="text-sm text-gray-400">Efficiency: {explanation.efficiency}%</p>
          </>
        ) : (
          <p>Run prediction to see insights</p>
        )}
      </motion.div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-xl bg-gray-900 p-4">
        <h2 className="mb-2 text-lg font-semibold">Decision Logic</h2>

        {typeof demand === "number" && typeof waste === "number" ? (
          <>
            <p>Demand: {demand.toFixed(2)}</p>
            <p>Waste: {waste.toFixed(2)}</p>
            <p>Risk Level: {risk}</p>
            <p className="mt-2 font-semibold">
              Recommended Production: {typeof production === "number" ? production.toFixed(2) : "--"}
            </p>

            <p className="mt-2 text-sm text-gray-400">
              System suggests adjusting production based on demand-waste imbalance.
            </p>
          </>
        ) : (
          <p>No decision available yet</p>
        )}
      </motion.div>
    </motion.div>
  )
}
