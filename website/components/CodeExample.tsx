import { Terminal, Copy } from "lucide-react";

export default function CodeExample() {
    return (
        <section className="bg-[#0f172a] py-24 sm:py-32 relative overflow-hidden">
            <div className="absolute inset-x-0 top-0 h-px bg-white/10" />
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl lg:text-center mb-16">
                    <h2 className="text-base font-semibold leading-7 text-emerald-400">Dead Simple CLI</h2>
                    <p className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">One command to rule them all</p>
                </div>

                <div className="rounded-xl overflow-hidden bg-[#1e293b] shadow-2xl border border-white/5 max-w-4xl mx-auto">
                    <div className="flex justify-between items-center px-4 py-3 bg-[#0f172a] border-b border-white/5">
                        <div className="flex gap-2">
                            <div className="w-3 h-3 rounded-full bg-red-500" />
                            <div className="w-3 h-3 rounded-full bg-yellow-500" />
                            <div className="w-3 h-3 rounded-full bg-green-500" />
                        </div>
                        <div className="text-xs text-slate-400 font-mono">bash</div>
                    </div>

                    <div className="p-6 font-mono text-sm overflow-x-auto">
                        <div className="flex items-center gap-2 mb-4 group cursor-pointer text-emerald-400">
                            <span className="text-pink-500">$</span>
                            <span>llmq eval --provider groq</span>
                            <Copy className="w-4 h-4 text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity ml-auto" />
                        </div>

                        <div className="text-slate-300 space-y-1">
                            <div className="opacity-50">Loading dataset: examples/customer_support.jsonl ...</div>
                            <div className="opacity-50">Initializing Groq provider (llama3-70b-8192) ...</div>
                            <br />
                            <div className="flex gap-2">
                                <span className="text-blue-400">RUNNING</span>
                                <span>Evaluation [====================] 100%</span>
                            </div>
                            <br />
                            <div className="text-emerald-400 font-bold">✔ PASSED (9/10 checks)</div>
                            <div>----------------------------------------</div>
                            <div className="grid grid-cols-2 gap-4 max-w-md">
                                <div>relevance_score:</div><div className="text-emerald-400">0.92</div>
                                <div>hallucination_rate:</div><div className="text-emerald-400">0.05</div>
                                <div>consistency:</div><div className="text-yellow-400">0.88</div>
                                <div>latency_p95:</div><div className="text-emerald-400">450ms</div>
                            </div>
                            <br />
                            <div className="text-slate-400">Report generated: ./reports/run_20240520_1432.html</div>
                            <div className="text-slate-400">Dashboard available at http://localhost:8000</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
