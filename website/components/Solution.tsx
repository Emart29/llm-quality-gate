import { CheckCircle, Blocks, GitMerge, Monitor } from "lucide-react";

const solutions = [
    {
        icon: CheckCircle,
        title: "Deterministic Checks",
        description: "Run evaluations across fixed datasets. Get Pass/Fail outcomes on semantic similarity and factual accuracy.",
        link: "#metrics"
    },
    {
        icon: Blocks,
        title: "Multi-Provider Agnostic",
        description: "Compare OpenAI vs Claude vs Llama. Switch providers in one config line without rewriting tests.",
        link: "#providers"
    },
    {
        icon: GitMerge,
        title: "Quality Gates in CI",
        description: "Block PRs if accuracy drops below 95%. Native integration with GitHub Actions and GitLab CI.",
        link: "#ci"
    },
    {
        icon: Monitor,
        title: "Local Debug Dashboard",
        description: "Visualize failure cases instantly. Inspect traces, prompt variations, and latency metrics locally.",
        link: "#dashboard"
    }
];

export default function Solution() {
    return (
        <section id="solution" className="py-24 sm:py-32 bg-background relative overflow-hidden">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl lg:text-center">
                    <h2 className="text-base font-semibold leading-7 text-indigo-600">The Solution</h2>
                    <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
                        Reliable foundations for AI apps
                    </p>
                    <p className="mt-6 text-lg leading-8 text-muted-foreground">
                        LLMQ brings software engineering rigor to prompt engineering. Stop guessing, start measuring.
                    </p>
                </div>

                <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
                    <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-4">
                        {solutions.map((feature) => (
                            <div key={feature.title} className="flex flex-col items-center text-center">
                                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-50 border border-indigo-100 dark:bg-indigo-900/20 dark:border-indigo-800 text-indigo-600">
                                    <feature.icon className="h-8 w-8" aria-hidden="true" />
                                </div>
                                <dt className="text-xl font-semibold leading-7 text-foreground">
                                    {feature.title}
                                </dt>
                                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-muted-foreground">
                                    <p className="flex-auto">{feature.description}</p>
                                </dd>
                            </div>
                        ))}
                    </dl>
                </div>
            </div>
        </section>
    );
}
