import Link from "next/link";
import { ArrowRight, Github } from "lucide-react";

export default function Hero() {
    return (
        <section className="relative pt-32 pb-20 overflow-hidden bg-background">
            <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
                <div
                    className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#3b82f6] to-[#9333ea] opacity-30 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
                    style={{ clipPath: "polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)" }}
                />
            </div>

            <div className="mx-auto max-w-7xl px-6 lg:px-8 text-center relative z-10">
                <div className="mx-auto max-w-2xl py-8">
                    <div className="mb-8 flex justify-center">
                        {/* Logo placeholder - using a styled element */}
                        <div className="relative inline-flex items-center justify-center p-4 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 shadow-lg shadow-blue-500/20">
                            <span className="text-4xl font-bold text-white tracking-widest font-mono">LLMQ</span>
                        </div>
                    </div>

                    <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-6xl mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 pb-2">
                        Regression Testing <br /> for LLMs
                    </h1>

                    <p className="mx-auto max-w-xl text-lg leading-8 text-muted-foreground mb-10">
                        Open-source quality gates for AI applications. Ensure deterministic outputs, catch silent regressions, and integrate seamlessly with your CI pipeline.
                    </p>

                    <div className="mt-10 flex items-center justify-center gap-x-6">
                        <Link
                            href="https://github.com/Emart29/llm-quality-gate"
                            target="_blank"
                            className="rounded-md bg-foreground px-5 py-3 text-sm font-semibold text-background shadow-sm hover:bg-foreground/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 flex items-center gap-2 transition-transform hover:scale-105 duration-200"
                        >
                            <Github className="w-5 h-5" />
                            View on GitHub
                        </Link>
                        <Link
                            href="#get-started"
                            className="text-sm font-semibold leading-6 text-foreground hover:text-blue-600 transition-colors flex items-center gap-2 group"
                        >
                            Get Started <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </Link>
                    </div>
                </div>
            </div>
        </section>
    );
}
