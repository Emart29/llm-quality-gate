import { Github, Twitter } from "lucide-react";
import Link from "next/link";

export default function Footer() {
    return (
        <footer className="bg-background py-16 border-t border-border">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="flex flex-col items-center justify-between gap-6 sm:flex-row">
                    <p className="text-sm text-muted-foreground">
                        &copy; {new Date().getFullYear()} LLMQ. Open source under MIT License.
                    </p>
                    <div className="flex gap-6">
                        <Link
                            href="https://github.com/Emart29/llm-quality-gate"
                            target="_blank"
                            className="text-muted-foreground hover:text-foreground transition-colors"
                        >
                            <span className="sr-only">GitHub</span>
                            <Github className="w-5 h-5" />
                        </Link>
                        <Link
                            href="https://x.com/Ema_rtN"
                            target="_blank"
                            className="text-muted-foreground hover:text-foreground transition-colors"
                        >
                            <span className="sr-only">X (Twitter)</span>
                            <Twitter className="w-5 h-5" />
                        </Link>
                    </div>
                </div>
                <div className="mt-8 flex justify-center gap-8 text-sm text-muted-foreground">
                    <Link href="https://github.com/Emart29/llm-quality-gate/blob/main/CONTRIBUTING.md" target="_blank" className="hover:underline hover:text-foreground">Contributing</Link>
                    <Link href="https://github.com/Emart29/llm-quality-gate/blob/main/LICENSE" target="_blank" className="hover:underline hover:text-foreground">License</Link>
                    <Link href="/privacy" className="hover:underline hover:text-foreground">Privacy</Link>
                </div>
            </div>
        </footer>
    );
}
