import {
    BarChart3,
    Layers,
    Server,
    Cpu,
    ShieldCheck,
    Terminal
} from "lucide-react";

const features = [
    {
        name: 'Unified Provider Interface',
        description: 'Switch between OpenAI, Anthropic, Gemini, Groq, and Hugging Face with zero code changes. Abstraction layer handles retry logic and rate limits.',
        icon: Server,
    },
    {
        name: 'Semantic Metrics',
        description: 'Beyond simple string matching. Uses embedding similarity (Cosine) and LLM-as-a-Judge to evaluate answer relevance and correctness.',
        icon: Layers,
    },
    {
        name: 'CI/CD Native',
        description: 'Designed for GitHub Actions and GitLab CI. Returns proper exit codes and JUnit XML reports for integration with existing pipelines.',
        icon: ShieldCheck,
    },
    {
        name: 'Local Dashboard',
        description: 'Inspect every run visually. Compare prompt versions side-by-side. Track latency and cost per token across all your providers.',
        icon: BarChart3,
    },
    {
        name: 'Task-Specific Metrics',
        description: 'Custom metrics for RAG (retrieval accuracy), Summarization (content preservation), and Code Generation (syntax validity).',
        icon: Cpu,
    },
    {
        name: 'Developer First',
        description: 'Fully typed Python SDK and CLI. Configurable via simple YAML. Extensible plugin system for custom metrics and judges.',
        icon: Terminal,
    },
];

export default function Features() {
    return (
        <div className="bg-background py-24 sm:py-32" id="features">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl lg:text-center">
                    <h2 className="text-base font-semibold leading-7 text-indigo-600">Everything you need</h2>
                    <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
                        Complete Toolkit for LLM Quality
                    </p>
                    <p className="mt-6 text-lg leading-8 text-muted-foreground">
                        From local debugging to production gates, LLMQ covers the entire lifecycle of your AI features.
                    </p>
                </div>
                <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
                    <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-3">
                        {features.map((feature) => (
                            <div key={feature.name} className="relative pl-16 group hover:translate-y-[-2px] transition-transform duration-200">
                                <dt className="text-base font-semibold leading-7 text-foreground">
                                    <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600 group-hover:bg-indigo-500 transition-colors">
                                        <feature.icon className="h-6 w-6 text-white" aria-hidden="true" />
                                    </div>
                                    {feature.name}
                                </dt>
                                <dd className="mt-2 text-base leading-7 text-muted-foreground">
                                    {feature.description}
                                </dd>
                            </div>
                        ))}
                    </dl>
                </div>
            </div>
        </div>
    );
}
