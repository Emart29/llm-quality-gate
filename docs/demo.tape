Output docs/demo.gif
Set FontSize 16
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms

# Title
Type "# ────────────────────────────────────────"
Enter
Type "#  LLMQ — LLM Quality Gate Demo (60s)"
Enter
Type "# ────────────────────────────────────────"
Enter
Sleep 2s

# Scene 1: Init
Type "# Scene 1: Initialize a new project"
Enter
Sleep 500ms
Type "llmq init"
Enter
Sleep 6s

# Scene 2: Eval (pass)
Type "# Scene 2: Run evaluation (expecting PASS)"
Enter
Sleep 500ms
Type "llmq eval --provider groq"
Enter
Sleep 18s

# Scene 3: Eval (fail — CI mode)
Type "# Scene 3: CI mode — quality gate enforcement"
Enter
Sleep 500ms
Type "llmq eval --provider groq --fail-on-gate"
Enter
Sleep 12s
Type "echo \"Exit code: $?\""
Enter
Sleep 2s

# Scene 4: Dashboard
Type "# Scene 4: Launch dashboard"
Enter
Sleep 500ms
Type "llmq dashboard &"
Enter
Sleep 3s
Type "echo \"→ Open http://localhost:8000 in your browser\""
Enter
Sleep 5s

Type "# ✅  Demo complete — see docs/PRESENTATION_ASSETS.md for more"
Enter
Sleep 3s
