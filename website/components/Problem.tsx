import { Shuffle, Search, AlertTriangle, EyeOff } from "lucide-react";

const problems = [
    {
        icon: Shuffle,
        title: "Nondeterministic Outputs",
        description: "LLMs are probabilistic. Even with temperature=0, outputs drift over time, breaking your app silently.",
    },
    {
        icon: EyeOff,
        title: "Silent Regressions",
        description: "A small prompt tweak improves one case but breaks 5 others. You won't know until users complain.",
    },
    {
        icon: Search,
        title: "No Standard CI/CD",
        description: "Traditional unit tests (assert 'foo' == 'bar') fail on semantic variations. You need fuzzy matching.",
    },
    {
        icon: AlertTriangle,
        title: "Hallucination Risks",
        description: "Models confidently generate false information. Catching this requires systematic fact-checking.",
    },
];

export default function Problem() {
    return (
        <section id="problem" className="py-24 bg-secondary/30 relative">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl text-center mb-16">
                    <h2 className="text-base font-semibold leading-7 text-blue-600">The Challenge</h2>
                    <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">Why testing LLMs is hard</p>
                    <p className="mt-6 text-lg leading-8 text-muted-foreground">
                        Building reliable AI applications requires a new approach to quality assurance. Traditional testing tools fall short.
                    </p>
                </div>

                <div className="mx-auto max-w-7xl grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12">
                    {problems.map((problem, idx) => (
                        <div
                            key={idx}
                            className="group relative flex items-start p-8 rounded-2xl bg-background border border-border shadow-sm hover:shadow-md transition-all duration-300 hover:border-blue-500/50"
                        >
                            <div className="flex-shrink-0 p-3 rounded-lg bg-blue-500/10 text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors duration-300">
                                <problem.icon className="h-6 w-6" aria-hidden="true" />
                            </div>
                            <div className="ml-6">
                                <h3 className="text-xl font-bold text-foreground mb-3 group-hover:text-blue-600 transition-colors">
                                    {problem.title}
                                </h3>
                                <p className="text-muted-foreground leading-relaxed">
                                    {problem.description}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
