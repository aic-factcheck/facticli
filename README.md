# facticli

An agentic CLI fact-checking framework built at the [AI Center, Czech Technical University in Prague](https://aic.fel.cvut.cz/).

**facticli** reimagines state-of-the-art academic fact-checking as a modular, agentic CLI tool. By moving beyond static RAG pipelines toward tool-augmented, multi-step reasoning, it explores what new capabilities emerge when a fact-checker can autonomously plan, retrieve, and verify claims through composable CLI actions.

## Key Ideas

- 🔍 **Agentic fact-checking** — the system autonomously decomposes claims, plans verification strategies, and iterates on evidence
- 🛠️ **CLI-native** — built around composable command-line tooling, not notebooks or web UIs
- 🔌 **Model-agnostic** — currently supports GPT-5.3, designed to swap in any LLM backend
- 📐 **Research-first** — developed as a research tool to probe the boundaries of agentic fact verification

## Getting Started

```bash
pip install facticli
facticli check "The Eiffel Tower was built in 1889 for the World's Fair."
```

## Citation

If you use facticli in your research, please cite:

```bibtex
@misc{ullrich2025facticli,
  title={facticli: ...},
  author={Ullrich, Herbert and ...},
  year={2025}
}
```

## License

cc-by-sa-4.0
