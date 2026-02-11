import { Copy } from "lucide-react";

export default function Documentation() {
    return (
        <section id="docs" className="py-24 bg-background">
            <div className="mx-auto max-w-5xl px-6 lg:px-8">
                <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl mb-12">Getting Started</h2>

                <div className="space-y-12">
                    {/* Installation */}
                    <div className="group">
                        <h3 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
                            <span className="text-blue-600">01.</span> Installation
                        </h3>
                        <div className="bg-slate-900 rounded-lg p-4 font-mono text-sm text-slate-200">
                            <span className="text-pink-500">$</span> pip install llmq-gate
                        </div>
                    </div>

                    {/* Configuration */}
                    <div>
                        <h3 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
                            <span className="text-blue-600">02.</span> Configuration
                        </h3>
                        <p className="text-muted-foreground mb-4">
                            Create a `llmq_config.yaml` file in your root directory.
                        </p>
                        <div className="bg-slate-900 rounded-lg p-4 font-mono text-sm text-slate-200 overflow-x-auto">
                            <pre>{`providers:
  openai:
    model: gpt-4-turbo
    api_key: \${OPENAI_API_KEY}

metrics:
  - name: correctness
    type: llm_judge
    threshold: 0.8
  - name: hallucination
    type: custom
    path: ./metrics/hallucination.py

dataset:
  path: ./data/eval_set.jsonl
  input_field: prompt
  reference_field: expected_completion`}</pre>
                        </div>
                    </div>

                    {/* CI Setup */}
                    <div>
                        <h3 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
                            <span className="text-blue-600">03.</span> CI Integration
                        </h3>
                        <p className="text-muted-foreground mb-4">
                            Add to your GitHub Actions workflow. Returns non-zero exit code on failure.
                        </p>
                        <div className="bg-slate-900 rounded-lg p-4 font-mono text-sm text-slate-200 overflow-x-auto">
                            <pre>{`- name: Run LLM Quality Gate
  run: llmq eval --ci
  env:
    OPENAI_API_KEY: \${{ secrets.OPENAI_API_KEY }}`}</pre>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
