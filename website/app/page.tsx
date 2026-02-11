import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Problem from "@/components/Problem";
import Solution from "@/components/Solution";
import CodeExample from "@/components/CodeExample";
import Features from "@/components/Features";
import Architecture from "@/components/Architecture";
import Documentation from "@/components/Documentation";
import Screenshots from "@/components/Screenshots";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navbar />
      <Hero />
      <Problem />
      <Solution />
      <CodeExample />
      <Features />
      <Architecture />
      <Documentation />
      <Screenshots />
      <Footer />
    </main>
  );
}
