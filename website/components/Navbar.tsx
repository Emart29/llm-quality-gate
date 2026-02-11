import Link from "next/link";
import { Github } from "lucide-react";

export default function Navbar() {
    return (
        <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-sm border-b border-border">
            <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2 font-bold text-xl">
                    <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-mono text-sm">
                        Q
                    </div>
                    LLMQ
                </Link>
                <div className="flex items-center gap-6">
                    <div className="hidden md:flex items-center gap-6 text-sm text-muted-foreground">
                        <Link href="#problem" className="hover:text-foreground transition-colors">Problem</Link>
                        <Link href="#solution" className="hover:text-foreground transition-colors">Solution</Link>
                        <Link href="#features" className="hover:text-foreground transition-colors">Features</Link>
                        <Link href="#docs" className="hover:text-foreground transition-colors">Docs</Link>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link
                            href="https://github.com/Emart29/llm-quality-gate"
                            target="_blank"
                            className="p-2 hover:bg-accent rounded-md transition-colors"
                        >
                            <Github className="w-5 h-5" />
                        </Link>
                        <Link
                            href="#get-started"
                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
                        >
                            Get Started
                        </Link>
                    </div>
                </div>
            </div>
        </nav>
    );
}
