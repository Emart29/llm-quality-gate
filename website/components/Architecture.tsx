

export default function Architecture() {
    return (
        <section className="py-24 sm:py-32 bg-secondary/20 relative items-center flex flex-col justify-center">
            <div className="mx-auto max-w-7xl px-6 lg:px-8 text-center">
                <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl mb-4">How It Works</h2>
                <p className="text-lg text-muted-foreground mb-12 max-w-2xl mx-auto">
                    The modular pipeline allows you to swap components at any stage. Bring your own dataset, custom judges, or integrate with your preferred Vector DB.
                </p>

                <div className="bg-white p-8 rounded-2xl shadow-xl overflow-hidden border border-border">
                    {/* Using next/image for optimization, or just standard img for svg if width/height tricky */}
                    <div className="relative w-full h-auto min-h-[400px] flex items-center justify-center">
                        <img
                            src="/architecture.svg"
                            alt="LLMQ Architecture Diagram"
                            className="max-w-full h-auto"
                        />
                    </div>
                </div>

                <div className="mt-12 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 text-sm font-medium text-muted-foreground">
                    {[
                        "Dataset", "→", "Generator", "→", "Judge", "→", "Metrics"
                    ].map((step, i) => (
                        <div key={i} className="flex items-center justify-center p-2 rounded bg-background border border-border shadow-sm">
                            {step}
                        </div>
                    ))}
                    {/* Wrap to next line on small screens or continue */}
                    <div className="hidden lg:flex items-center justify-center text-muted-foreground">→</div>
                    <div className="col-span-2 md:col-span-4 lg:col-span-2 flex items-center justify-center p-2 rounded bg-background border border-border shadow-sm">
                        Quality Gate
                    </div>
                    {/* etc. Simplified flow here for visual cleanliness */}
                </div>
            </div>
        </section>
    );
}
