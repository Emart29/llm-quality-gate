export default function Screenshots() {
    return (
        <section className="py-24 sm:py-32 bg-secondary/20" id="dashboard">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl lg:text-center mb-16">
                    <h2 className="text-base font-semibold leading-7 text-indigo-600">Visual Insights</h2>
                    <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
                        Beautiful Local Dashboard
                    </p>
                    <p className="mt-6 text-lg leading-8 text-muted-foreground">
                        Track performance over time, compare model versions, and drill down into individual failures.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                    <div className="rounded-xl overflow-hidden shadow-2xl border border-border bg-background p-2">
                        <img
                            src="/images/evaluation-summary.svg"
                            alt="Evaluation Summary Dashboard"
                            className="w-full h-auto rounded-lg"
                        />
                        <p className="text-center text-sm text-muted-foreground mt-2 font-mono">Run Summary View</p>
                    </div>

                    <div className="rounded-xl overflow-hidden shadow-xl border border-border bg-background p-2">
                        <img
                            src="/images/historical-runs.svg"
                            alt="Historical Performance Trends"
                            className="w-full h-auto rounded-lg"
                        />
                        <p className="text-center text-sm text-muted-foreground mt-2 font-mono">Historical Trends</p>
                    </div>
                </div>

                <div className="mt-16 rounded-xl overflow-hidden shadow-2xl border border-border bg-background p-2 max-w-4xl mx-auto">
                    <img
                        src="/images/model-comparison.svg"
                        alt="Model Comparison Chart"
                        className="w-full h-auto rounded-lg"
                    />
                    <p className="text-center text-sm text-muted-foreground mt-2 font-mono">Multi-Model Comparison</p>
                </div>
            </div>
        </section>
    );
}
